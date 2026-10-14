# Game patches: the patch file format and applying the selected patches to the
# main module before it runs. Depends on no other delta module; the launcher
# installs it as the kernel's main-image hook.
add_delta_module(patch)
