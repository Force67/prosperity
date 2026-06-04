package com.prosperity.ps4;

/**
 * Thin bridge to the native utilities exported by libps4delta_app.so
 * (delta/main/android_jni.cpp). Loaded in the launcher process only to read pkg
 * metadata and unpack firmware; the emulator itself is entered via
 * NativeActivity in a separate process.
 */
public final class NativeBridge {
    static {
        try {
            System.loadLibrary("ps4delta_app");
            android.util.Log.i("prosperity", "native lib loaded");
        } catch (Throwable t) {
            android.util.Log.e("prosperity", "loadLibrary(ps4delta_app) failed", t);
            throw t;
        }
    }

    private NativeBridge() {}

    /** "<TITLE_ID>\t<TITLE>" from the pkg's param.sfo, or "" if unreadable. */
    public static native String pkgInfo(String pkgPath);

    /** Extract /sce_sys/icon0.png from the pkg to outPath. */
    public static native boolean pkgIcon(String pkgPath, String outPath);

    /** Best-effort unpack a firmware .PUP into outDir; returns a status summary. */
    public static native String pupExtract(String pupPath, String outDir);
}
