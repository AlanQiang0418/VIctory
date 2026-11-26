# D16 RGBA16_UNORM samples with a runtime sampler heap

On the #2030 author revision `e87c4f1f03ec92794c3008a75c8ef0371b70cf87`, point samples of stored codes `0xffaf` and `0xffef` each rounded one binary16 step above the exact normalized texel. The RTX 4090 Laptop GPU (10de:2757, driver 621.14) and WARP (1414:008c, d3d10warp.dll 10.0.26100.10068) agreed on every D16 result. [Author PR](https://github.com/Marvisak/VIctory/pull/2030)

| Left/right red channel codes | Sampler | RTX f32 bits | WARP f32 bits | D16 bits, both devices | Reference D16 |
| --- | --- | --- | --- | --- | --- |
| `0000 / ffaf` | point | `3f7fb000` | `3f7fb000` | `3bfe` | `3bfd` |
| `0000 / ffaf` | linear | `3effb100` | `3effb000` | `37fe` | `37fe` |
| `0000 / ffef` | point | `3f7ff000` | `3f7ff000` | `3c00` | `3bff` |
| `0000 / ffef` | linear | `3efff100` | `3efff000` | `3800` | `3800` |
| `ff6f / ffef` | point | `3f7ff000` | `3f7ff000` | `3c00` | `3bff` |
| `ff6f / ffef` | linear | `3f7fb000` | `3f7fb000` | `3bfe` | `3bfe` |

The point reference uses integer distances to round the exact fraction `code / 65535` once to the nearest binary16, breaking ties to even. Prior native gfx1036 measurements cover point samples and loads of 256 sixteen-bit normalized codes. The two point reference values above are calculated from that rule. For linear samples the reference rounds the paired f32 returned by the same device, so its interpolation precision is included. [Native gfx1036 measurements](https://github.com/boykopovar/AnyPS5/pull/22750)

The last linear case and the first point case produce the same f32 bits on both devices. Their reference D16 values differ. An adjustment based solely on those f32 bits cannot satisfy both references.

## Reproduce

On Windows with the SDK's x64 `dxc.exe` and `dxil.dll`, run this from the checkout. Adjust the SDK version to match the installation.

```powershell
$sdk = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64'
$env:VICTORY_PC_DXC = Join-Path $sdk 'dxc.exe'
$env:VICTORY_DXIL = Join-Path $sdk 'dxil.dll'
cargo test --locked -p victory-shader --lib measure_heap_sampler_d16_unorm -- --ignored --nocapture --test-threads=1
```

The opt-in fixture translates one synthetic GFX10 compute shader. Each dispatch samples a 2-by-1 texture at (0.5, 0.5), once with D16 and once with f32, and writes both words to the same UAV. The DXIL check follows both sampler handles from `createHandleFromHeap` with the sampler-heap flag through `annotateHandle` into `sampleLevel`. The shader declares zero static samplers. [Fixture source](../crates/victory-shader/src/emit/synthetic_tests/runtime_sampler.rs)

Each device uses one pipeline and one persistent shader-visible sampler heap. Slot 0 stays point. Slot 1 changes from point to linear and back to point after each completed dispatch. The translated shader reads slot 1 through its sampler table. For each texture both point visits returned identical bits. Three separate test processes produced identical readbacks on both adapters. The fixture prints the point comparison, checks the linear reference and verifies the return to point. A successful test run means the measurements and controls completed.
