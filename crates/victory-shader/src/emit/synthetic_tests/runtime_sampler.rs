//! Opt-in measurement for #2030. A single guest shader samples the same
//! coordinates into a D16 word and a companion f32 word in one dispatch.

use super::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_UNORM;
use windows::Win32::Graphics::Dxgi::*;

fn sampling_shader() -> HostShader {
    let code: [u32; 24] = [
        0xBE84_0300,
        0xBE85_0380, // s[4:5] = view table (s0)
        0xBE86_0301,
        0xBE87_0380, // s[6:7] = sampler table (s1)
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0 (texture)
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 64 (UAV)
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0 (sampler)
        0x7E00_02F0,
        0x7E02_02F0, // v0 = v1 = 0.5
        0x7E08_0280, // v4 = 0 (output byte offset)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0108,
        0x8043_0200, // image_sample_lz v2, v[0:1], s[12:19], s[8:11] d16
        0xF09C_0108,
        0x0043_0300, // image_sample_lz v3, v[0:1], s[12:19], s[8:11]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0204, // buffer_store_dword v2, v4, s[40:43], offset:0
        0xE070_1004,
        0x800A_0304, // buffer_store_dword v3, v4, s[40:43], offset:4
        0xBF81_0000, // s_endpgm
    ];
    let table = |user_data| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let resource = |kind, user_data, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    };
    build_native(&NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![table(0), table(1)],
        used: vec![
            resource(ResourceType::Srv, 0, 0, 8),
            resource(ResourceType::Uav, 0, 16, 4),
            resource(ResourceType::Sampler, 1, 0, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    })
    .expect("lower heap sampler")
}

// A nonnegative binary16 up to 1 is an integer in units of 2^-24.
// Compare to code / 65535 using integer distances and break ties to even.
fn exact_unorm16(code: u16) -> u16 {
    let numerator = u64::from(code) << 24;
    (0..=0x3c00u16)
        .min_by_key(|&bits| {
            let exponent = bits >> 10;
            let mantissa = u64::from(bits & 0x3ff);
            let units = if exponent == 0 {
                mantissa
            } else {
                (1024 + mantissa) << (exponent - 1)
            };
            ((65535 * units).abs_diff(numerator), bits & 1)
        })
        .unwrap()
}

#[test]
#[ignore = "measurement for #2030; requires dxc, a hardware SM 6.6 device, and WARP"]
fn measure_heap_sampler_d16_unorm() {
    let shader = sampling_shader();
    assert!(shader.static_samplers.is_empty() && shader.samplers.is_empty());
    let assembly =
        super::super::tests::dump_dxil(&shader.dxil).expect("set VICTORY_PC_DXC to dxc.exe");
    let sampler_handles: Vec<_> = assembly
        .lines()
        .filter(|line| {
            line.contains("call %dx.types.Handle @dx.op.createHandleFromHeap")
                && line.contains("i1 true, i1 false")
        })
        .collect();
    assert_eq!(
        sampler_handles.len(),
        2,
        "D16 and f32 must both use the sampler heap"
    );
    for handle in sampler_handles {
        let operand = |line: &str| {
            format!(
                "%dx.types.Handle {},",
                line.split_whitespace().next().unwrap()
            )
        };
        let annotation = assembly
            .lines()
            .find(|line| {
                line.contains("call %dx.types.Handle @dx.op.annotateHandle")
                    && line.contains(&operand(handle))
            })
            .expect("annotate the sampler-heap handle");
        let sample = assembly
            .lines()
            .find(|line| {
                line.contains("call %dx.types.ResRet.f32 @dx.op.sampleLevel.f32")
                    && line.contains(&operand(annotation))
            })
            .expect("SampleLevel must consume the annotated sampler-heap handle");
        for line in [handle, annotation, sample] {
            eprintln!("DXIL {}", line.trim());
        }
    }
    let mut constants = vec![0; shader.user_data_dwords() as usize];
    for arg in &shader.user_data {
        assert_eq!(arg.root, RootType::Table);
        constants[arg.index as usize] = match arg.slot {
            0 => 0,  // view heap slot zero
            1 => 16, // one 16-byte guest sampler descriptor: sampler heap slot one
            other => panic!("unexpected root slot {other}"),
        };
    }
    let point = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    let linear = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        ..point
    };
    let format = DXGI_FORMAT_R16G16B16A16_UNORM;
    let _one = super::super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for warp in [false, true] {
        let host = Host::new(warp).expect("SM 6.6 device");
        // SAFETY: query the actual device, then allocate a heap it owns.
        let (adapter, heap) = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapterByLuid(host.device.GetAdapterLuid())
                .unwrap();
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .unwrap();
            (adapter.GetDesc1().unwrap(), heap)
        };
        eprintln!(
            "adapter {:04x}:{:04x} {}",
            adapter.VendorId,
            adapter.DeviceId,
            String::from_utf16_lossy(&adapter.Description).trim_end_matches('\0')
        );
        assert_eq!(
            adapter.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
            warp
        );
        let (root, pso) = host
            .compute_pipeline(&shader)
            .expect("one pipeline for every filter");
        let mut point_at_ffaf = None;
        for (left, code, reference) in [
            (0, 0xffaf, 0x3bfd),
            (0, 0xffef, 0x3bff),
            (0xff6f, 0xffef, 0x3bff),
        ] {
            let exact = exact_unorm16(code);
            assert_eq!(exact, reference);
            // At u=0.5, point selects the right texel; linear mixes left and right.
            let texels: Vec<_> = [left, code]
                .into_iter()
                .flat_map(|c| [c; 4].into_iter().flat_map(u16::to_le_bytes))
                .collect();
            let views = [
                View::TextureBits {
                    size: [2, 1],
                    format,
                    data: &texels,
                    texel_bytes: 8,
                },
                View::Buffer {
                    data: &[0; 4],
                    stride: Stride::Structured(16),
                },
                View::RawUav(2),
            ];
            let mut first_point = None;
            for (name, descriptor) in [("point", point), ("linear", linear), ("point", point)] {
                let result = host
                    .dispatch_with_sampler_heap(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![
                                (0, Root::Constants(constants.clone())),
                                (
                                    1,
                                    Root::View {
                                        srv: true,
                                        data: vec![
                                            super::super::host::texture_rounding(format),
                                            16,
                                            4,
                                        ],
                                    },
                                ),
                            ],
                            views: &views,
                            samplers: &[point, descriptor],
                            groups: [1, 1, 1],
                        },
                        Some(&heap),
                    )
                    .expect("dispatch and fence before rewriting the same sampler slot");
                let [bits, fbits] = result[0][..] else {
                    panic!("expected two words")
                };
                let f = f32::from_bits(fbits);
                let paired = super::super::semantics_tests::to_f16(f64::from(f), false);
                eprintln!(
                    "texels=[{left:#06x},{code:#06x}] {name} f32={fbits:#010x} d16={bits:#06x} nearest_of_f32={paired:#06x} exact_point={exact:#06x}"
                );
                if name == "linear" {
                    assert!(
                        f > f32::from(left) / 65535.0 && f < f32::from(code) / 65535.0,
                        "expected a blend of the two texels"
                    );
                    assert_eq!(bits, paired, "linear sample");
                    if left == 0xff6f {
                        eprintln!(
                            "point(0xffaf) and linear(0xff6f,0xffef) returned identical f32 bits: {}",
                            point_at_ffaf == Some(fbits)
                        );
                    }
                } else {
                    assert_eq!(
                        (f * 65535.0).round_ties_even() as u16,
                        code,
                        "point must select the right-hand texel"
                    );
                    if let Some(first) = first_point {
                        assert_eq!(first, (fbits, bits))
                    }
                    first_point = Some((fbits, bits));
                    if code == 0xffaf {
                        point_at_ffaf = Some(fbits);
                    }
                }
            }
        }
    }
}
