# Bloodborne final composite
Status: draft

## Context
Bloodborne reaches its final fullscreen composite, where the pixel shader samples
the rendered scene and a three-layer 1025x1 R32F LUT at guest address
0x80e2151000. The scene binding resolves to a live render target, but the LUT
currently falls through the guest-texture path without a Vulkan image view. The
resulting scanout is red or black instead of the Brightness screen shown by the
source render target.

## Decision
Represent GCN 1D images as height-one Vulkan 2D images, and 1D-array images as
height-one Vulkan 2D-array images. Preserve the descriptor's layer count and
component mapping during layout creation, upload, view creation, and sampler
binding. Translate the recompiled shader's 1D coordinates as x plus an explicit
array layer, with the row coordinate fixed at the texel centre. Do not add a
Bloodborne-specific color or post-processing bypass.

## Alternatives
- Force the source render target to scan out: rejected because it skips the
  game's final color transform.
- Replace the LUT with a constant white or identity texture: rejected because it
  hides descriptor and shader errors and is not correct for other LUT contents.
- Special-case the Bloodborne shader address: rejected because the resource
  dimension is the renderer's reusable boundary.

## Consequences
1D resources use the existing 2D upload and descriptor machinery, with no new
Vulkan image type. The recompiler and texture cache must agree on the same
binding dimensionality, and the final pass remains dependent on correct guest
memory layout decoding.

## Acceptance
- The final composite binds the LUT as a non-null guest texture view.
- A captured post-composite frame contains the Brightness screen or later
  Bloodborne content without the red tiled pattern or black output.
- The existing `ps4delta` target builds successfully.
- Existing non-1D texture paths remain unchanged.
