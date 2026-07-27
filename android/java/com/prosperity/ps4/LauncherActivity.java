package com.prosperity.ps4;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Prosperity launcher: a game library backed by pkg files anywhere on the phone
 * (read in place by the native loader via All Files Access), plus a firmware
 * installer. Launching a game writes its path to boot.cfg and starts the
 * emulator (NativeActivity) in the :emu process.
 */
public class LauncherActivity extends Activity {

    private LinearLayout gamesContainer;
    private View firmwareBanner; // shown on the main page only when no firmware

    // --- lifecycle ----------------------------------------------------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        if (!hasAllFilesAccess())
            promptAllFilesAccess();
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshFirmwareStatus();
        rebuildGameList();
    }

    // --- UI construction ----------------------------------------------------

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(0xFF101418);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setLayoutParams(new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        // Top bar: title on the left, a Settings entry on the right.
        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setGravity(Gravity.CENTER_VERTICAL);
        topBar.setPadding(dp(16), dp(16), dp(8), dp(8));

        ImageView logo = new ImageView(this);
        int logoId = getResources().getIdentifier("logo", "drawable", getPackageName());
        if (logoId != 0)
            logo.setImageResource(logoId);
        LinearLayout.LayoutParams logoLp = new LinearLayout.LayoutParams(dp(36), dp(36));
        logoLp.rightMargin = dp(10);
        logo.setLayoutParams(logoLp);
        topBar.addView(logo);

        TextView title = new TextView(this);
        title.setText("Prosperity");
        title.setTextColor(Color.WHITE);
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setLayoutParams(new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        topBar.addView(title);

        Button settings = new Button(this);
        settings.setText("Settings");
        settings.setOnClickListener(v -> openSettings());
        topBar.addView(settings);
        content.addView(topBar);

        // First-run nudge: only on screen while no firmware is installed.
        content.addView(firmwareBanner = makeFirmwareBanner());

        View divider = new View(this);
        divider.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(1)));
        divider.setBackgroundColor(0xFF2A2F36);
        content.addView(divider);

        ScrollView scroll = new ScrollView(this);
        scroll.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        gamesContainer = new LinearLayout(this);
        gamesContainer.setOrientation(LinearLayout.VERTICAL);
        // Leave room so the FAB never covers the last row.
        gamesContainer.setPadding(0, 0, 0, dp(96));
        scroll.addView(gamesContainer);
        content.addView(scroll);

        root.addView(content);
        root.addView(makeFab());

        setContentView(root);
    }

    // Circular "+" floating action button, bottom-right, for adding a game.
    private View makeFab() {
        TextView fab = new TextView(this);
        fab.setText("+");
        fab.setTextColor(Color.WHITE);
        fab.setTextSize(30);
        fab.setTypeface(Typeface.DEFAULT_BOLD);
        fab.setGravity(Gravity.CENTER);
        fab.setIncludeFontPadding(false);
        GradientDrawable circle = new GradientDrawable();
        circle.setShape(GradientDrawable.OVAL);
        circle.setColor(0xFF3B82F6);
        fab.setBackground(circle);
        fab.setElevation(dp(6));
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(dp(60), dp(60));
        lp.gravity = Gravity.BOTTOM | Gravity.END;
        lp.setMargins(0, 0, dp(24), dp(28));
        fab.setLayoutParams(lp);
        fab.setOnClickListener(v -> addGame());
        return fab;
    }

    // "No firmware installed" card with an install button; hidden once present.
    private View makeFirmwareBanner() {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(16), dp(14), dp(16), dp(14));
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(0xFF1B2330);
        bg.setCornerRadius(dp(10));
        card.setBackground(bg);
        LinearLayout.LayoutParams clp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        clp.setMargins(dp(16), dp(8), dp(16), dp(8));
        card.setLayoutParams(clp);

        TextView head = new TextView(this);
        head.setText("Firmware not installed");
        head.setTextColor(Color.WHITE);
        head.setTextSize(16);
        head.setTypeface(Typeface.DEFAULT_BOLD);
        card.addView(head);

        TextView body = new TextView(this);
        body.setText("Install the PS4 firmware module set before running games.");
        body.setTextColor(0xFF9AA3AD);
        body.setTextSize(13);
        body.setPadding(0, dp(2), 0, dp(10));
        card.addView(body);

        Button install = new Button(this);
        install.setText("Install firmware");
        install.setOnClickListener(v -> installFirmware());
        card.addView(install);
        return card;
    }

    private void rebuildGameList() {
        if (gamesContainer == null)
            return;
        gamesContainer.removeAllViews();
        JSONArray lib = loadLibrary();
        if (lib.length() == 0) {
            TextView hint = new TextView(this);
            hint.setText("No games yet. Tap the + button and pick a PS4 .pkg, "
                    + "a PS5 .ffpkg, or an extracted PS5 game folder.");
            hint.setTextColor(0xFF6A7178);
            hint.setPadding(dp(16), dp(24), dp(16), dp(16));
            gamesContainer.addView(hint);
            return;
        }
        for (int i = 0; i < lib.length(); i++) {
            JSONObject g = lib.optJSONObject(i);
            if (g != null)
                gamesContainer.addView(makeGameRow(g));
        }
    }

    private View makeGameRow(JSONObject g) {
        final String path = g.optString("path");
        final String title = g.optString("title");
        String titleId = g.optString("titleId");
        String icon = g.optString("icon");

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(16), dp(12), dp(16), dp(12));
        row.setClickable(true);

        ImageView iv = new ImageView(this);
        LinearLayout.LayoutParams ip = new LinearLayout.LayoutParams(dp(56), dp(56));
        ip.rightMargin = dp(12);
        iv.setLayoutParams(ip);
        Bitmap bmp = icon.isEmpty() ? null : BitmapFactory.decodeFile(icon);
        if (bmp != null)
            iv.setImageBitmap(bmp);
        else
            iv.setBackgroundColor(0xFF2A2F36);
        row.addView(iv);

        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        TextView t = new TextView(this);
        t.setText(title.isEmpty() ? new File(path).getName() : title);
        t.setTextColor(Color.WHITE);
        t.setTextSize(16);
        t.setTypeface(Typeface.DEFAULT_BOLD);
        TextView s = new TextView(this);
        s.setText(titleId.isEmpty() ? path : (titleId + "   " + new File(path).getName()));
        s.setTextColor(0xFF7A828A);
        s.setTextSize(12);
        col.addView(t);
        col.addView(s);
        row.addView(col);

        row.setOnClickListener(v -> launchGame(path));
        row.setOnLongClickListener(v -> {
            confirmRemove(path, title);
            return true;
        });
        return row;
    }

    // --- game library actions ----------------------------------------------

    private void addGame() {
        if (!requireAccess())
            return;
        // PS4 titles are .pkg; PS5 ones are a .ffpkg backup or an extracted app
        // directory (eboot.bin + sce_sys), so folders are pickable too.
        browse(Environment.getExternalStorageDirectory(),
               new String[]{".pkg", ".ffpkg"}, true, f -> {
            runBusy("Reading " + f.getName() + " ...", () -> {
                String titleId = "";
                String title = "";
                String iconPath = "";
                // Title and cover art come from the outer PKG entry table, which
                // only a .pkg has; the other layouts fall back to their name.
                if (f.isFile() && f.getName().toLowerCase().endsWith(".pkg")) {
                    String info = safePkgInfo(f.getAbsolutePath());
                    if (info != null && info.indexOf('\t') >= 0) {
                        String[] parts = info.split("\t", 2);
                        titleId = parts[0];
                        title = parts.length > 1 ? parts[1] : "";
                    }
                    File iconsDir = new File(getFilesDir(), "icons");
                    iconsDir.mkdirs();
                    File iconFile = new File(iconsDir,
                            Integer.toHexString(f.getAbsolutePath().hashCode()) + ".png");
                    try {
                        if (NativeBridge.pkgIcon(f.getAbsolutePath(), iconFile.getAbsolutePath()))
                            iconPath = iconFile.getAbsolutePath();
                    } catch (Throwable ignored) {
                    }
                }
                if (title.isEmpty())
                    title = f.getName();
                addToLibrary(f.getAbsolutePath(), title, titleId, iconPath);
                return "Added " + title;
            });
        });
    }

    private void confirmRemove(final String path, String title) {
        new AlertDialog.Builder(this)
                .setTitle("Remove from library?")
                .setMessage(title)
                .setPositiveButton("Remove", (d, w) -> {
                    removeFromLibrary(path);
                    rebuildGameList();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void launchGame(String path) {
        if (!requireAccess())
            return;
        File f = new File(path);
        if (!f.exists()) {
            toast("File not found: " + path);
            return;
        }
        if (!firmwareInstalled()) {
            toast("Install firmware first");
            return;
        }
        File cfg = new File(getExternalFilesDir(null), "boot.cfg");
        try (FileWriter w = new FileWriter(cfg)) {
            w.write(path);
        } catch (Exception e) {
            toast("Cannot write boot.cfg: " + e.getMessage());
            return;
        }
        Intent i = new Intent();
        i.setClassName(this, "android.app.NativeActivity");
        startActivity(i);
    }

    private String safePkgInfo(String path) {
        try {
            return NativeBridge.pkgInfo(path);
        } catch (Throwable t) {
            android.util.Log.e("prosperity", "pkgInfo failed", t);
            return "";
        }
    }

    // --- firmware -----------------------------------------------------------

    private File modulesDir() {
        return new File(getExternalFilesDir(null), "modules");
    }

    private int moduleCount() {
        File[] files = modulesDir().listFiles();
        int count = 0;
        if (files != null)
            for (File f : files) {
                String n = f.getName().toLowerCase();
                if (n.endsWith(".sprx") || n.endsWith(".prx"))
                    count++;
            }
        return count;
    }

    private boolean firmwareInstalled() {
        return moduleCount() > 0;
    }

    private void refreshFirmwareStatus() {
        if (firmwareBanner != null)
            firmwareBanner.setVisibility(firmwareInstalled() ? View.GONE : View.VISIBLE);
    }

    // Settings menu: firmware lives here once installed (the main page only nags
    // about it while it's missing, via the banner).
    private void openSettings() {
        int count = moduleCount();
        String status = count > 0
                ? "Firmware: installed (" + count + " modules)"
                : "Firmware: not installed";
        new AlertDialog.Builder(this)
                .setTitle("Settings")
                .setMessage(status)
                .setPositiveButton("Install firmware", (d, w) -> installFirmware())
                .setNegativeButton("Close", null)
                .show();
    }

    private void installFirmware() {
        if (!requireAccess())
            return;
        String[] opts = {
                "From .zip of modules",
                "From a folder of .sprx",
                "From firmware .PUP (best effort)"};
        new AlertDialog.Builder(this)
                .setTitle("Install firmware")
                .setItems(opts, (d, w) -> {
                    File root = Environment.getExternalStorageDirectory();
                    if (w == 0)
                        browse(root, new String[]{".zip"}, false, this::installFromZip);
                    else if (w == 1)
                        browse(root, null, true, this::installFromFolder);
                    else
                        browse(root, new String[]{".pup"}, false, this::installFromPup);
                })
                .show();
    }

    private void installFromZip(File zip) {
        runBusy("Installing firmware from " + zip.getName() + " ...", () -> {
            File modules = modulesDir();
            modules.mkdirs();
            int n = 0;
            byte[] buf = new byte[1 << 16];
            try (ZipInputStream zis = new ZipInputStream(
                    new BufferedInputStream(new FileInputStream(zip)))) {
                ZipEntry e;
                while ((e = zis.getNextEntry()) != null) {
                    if (e.isDirectory())
                        continue;
                    String base = new File(e.getName()).getName();
                    if (base.isEmpty())
                        continue;
                    try (OutputStream os = new BufferedOutputStream(
                            new FileOutputStream(new File(modules, base)))) {
                        int r;
                        while ((r = zis.read(buf)) > 0)
                            os.write(buf, 0, r);
                    }
                    n++;
                }
            } catch (Exception ex) {
                return "Failed: " + ex.getMessage();
            }
            return "Installed " + n + " files into modules/";
        });
    }

    private void installFromFolder(File dir) {
        runBusy("Copying modules from " + dir.getName() + " ...", () -> {
            File modules = modulesDir();
            modules.mkdirs();
            File[] files = dir.listFiles();
            if (files == null)
                return "Cannot read folder";
            int n = 0;
            byte[] buf = new byte[1 << 16];
            for (File f : files) {
                if (!f.isFile())
                    continue;
                String nm = f.getName().toLowerCase();
                if (!(nm.endsWith(".sprx") || nm.endsWith(".prx")))
                    continue;
                try (InputStream is = new BufferedInputStream(new FileInputStream(f));
                     OutputStream os = new BufferedOutputStream(
                             new FileOutputStream(new File(modules, f.getName())))) {
                    int r;
                    while ((r = is.read(buf)) > 0)
                        os.write(buf, 0, r);
                } catch (Exception ex) {
                    return "Failed on " + f.getName() + ": " + ex.getMessage();
                }
                n++;
            }
            return n == 0 ? "No .sprx modules found in that folder"
                    : "Copied " + n + " modules into modules/";
        });
    }

    private void installFromPup(File pup) {
        runBusy("Unpacking " + pup.getName() + " ...", () -> {
            File outDir = new File(getExternalFilesDir(null), "pup");
            outDir.mkdirs();
            try {
                return NativeBridge.pupExtract(pup.getAbsolutePath(), outDir.getAbsolutePath());
            } catch (Throwable t) {
                return "PUP extract failed: " + t;
            }
        });
    }

    // --- persistence --------------------------------------------------------

    private SharedPreferences prefs() {
        return getSharedPreferences("prosperity", MODE_PRIVATE);
    }

    private JSONArray loadLibrary() {
        try {
            return new JSONArray(prefs().getString("games", "[]"));
        } catch (Exception e) {
            return new JSONArray();
        }
    }

    private synchronized void addToLibrary(String path, String title, String titleId, String icon) {
        try {
            JSONArray in = loadLibrary();
            JSONArray out = new JSONArray();
            for (int i = 0; i < in.length(); i++) {
                JSONObject o = in.getJSONObject(i);
                if (!path.equals(o.optString("path")))
                    out.put(o);
            }
            JSONObject g = new JSONObject();
            g.put("path", path);
            g.put("title", title);
            g.put("titleId", titleId);
            g.put("icon", icon);
            out.put(g);
            prefs().edit().putString("games", out.toString()).apply();
        } catch (Exception ignored) {
        }
    }

    private synchronized void removeFromLibrary(String path) {
        try {
            JSONArray in = loadLibrary();
            JSONArray out = new JSONArray();
            for (int i = 0; i < in.length(); i++) {
                JSONObject o = in.getJSONObject(i);
                if (!path.equals(o.optString("path")))
                    out.put(o);
            }
            prefs().edit().putString("games", out.toString()).apply();
        } catch (Exception ignored) {
        }
    }

    // --- All Files Access ---------------------------------------------------

    private boolean hasAllFilesAccess() {
        if (Build.VERSION.SDK_INT >= 30)
            return Environment.isExternalStorageManager();
        return checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                == android.content.pm.PackageManager.PERMISSION_GRANTED;
    }

    private void promptAllFilesAccess() {
        new AlertDialog.Builder(this)
                .setTitle("Storage access needed")
                .setMessage("Prosperity needs access to all files so it can open "
                        + "pkg files and firmware anywhere on your phone.")
                .setPositiveButton("Grant", (d, w) -> requestAllFilesAccess())
                .setNegativeButton("Later", null)
                .show();
    }

    private void requestAllFilesAccess() {
        if (Build.VERSION.SDK_INT >= 30) {
            try {
                Intent i = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                i.setData(Uri.parse("package:" + getPackageName()));
                startActivity(i);
            } catch (Exception e) {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            }
        } else {
            requestPermissions(
                    new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
    }

    private boolean requireAccess() {
        if (hasAllFilesAccess())
            return true;
        promptAllFilesAccess();
        return false;
    }

    // --- file browser -------------------------------------------------------

    private interface OnPick {
        void pick(File f);
    }

    private boolean matches(String name, String[] exts) {
        if (exts == null)
            return false; // file-pick disabled when only dirs are wanted
        String low = name.toLowerCase();
        for (String e : exts)
            if (low.endsWith(e))
                return true;
        return false;
    }

    private void browse(final File dir, final String[] exts,
                        final boolean allowDirPick, final OnPick cb) {
        final ArrayList<File> entries = new ArrayList<>();
        final ArrayList<String> names = new ArrayList<>();
        File parent = dir.getParentFile();
        if (parent != null) {
            entries.add(parent);
            names.add(".. (up)");
        }
        File[] all = dir.listFiles();
        if (all != null) {
            Arrays.sort(all, new Comparator<File>() {
                @Override
                public int compare(File a, File b) {
                    if (a.isDirectory() != b.isDirectory())
                        return a.isDirectory() ? -1 : 1;
                    return a.getName().compareToIgnoreCase(b.getName());
                }
            });
            for (File f : all) {
                if (f.isDirectory()) {
                    entries.add(f);
                    names.add("[ " + f.getName() + " ]");
                } else if (matches(f.getName(), exts)) {
                    entries.add(f);
                    names.add(f.getName());
                }
            }
        }

        AlertDialog.Builder b = new AlertDialog.Builder(this);
        b.setTitle(dir.getAbsolutePath());
        b.setItems(names.toArray(new String[0]), (d, which) -> {
            File sel = entries.get(which);
            if (sel.isDirectory())
                browse(sel, exts, allowDirPick, cb);
            else
                cb.pick(sel);
        });
        if (allowDirPick)
            b.setPositiveButton("Use this folder", (d, w) -> cb.pick(dir));
        b.setNegativeButton("Cancel", null);
        b.show();
    }

    // --- busy dialog + worker ----------------------------------------------

    private interface Work {
        String run();
    }

    private void runBusy(String message, final Work work) {
        final LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.HORIZONTAL);
        box.setGravity(Gravity.CENTER_VERTICAL);
        box.setPadding(dp(24), dp(24), dp(24), dp(24));
        ProgressBar pb = new ProgressBar(this);
        LinearLayout.LayoutParams pp = new LinearLayout.LayoutParams(dp(40), dp(40));
        pp.rightMargin = dp(20);
        pb.setLayoutParams(pp);
        box.addView(pb);
        TextView msg = new TextView(this);
        msg.setText(message);
        box.addView(msg);

        final AlertDialog busy = new AlertDialog.Builder(this)
                .setView(box).setCancelable(false).create();
        busy.show();

        new Thread(() -> {
            String result;
            try {
                result = work.run();
            } catch (Throwable t) {
                result = "Error: " + t;
            }
            final String r = result;
            runOnUiThread(() -> {
                busy.dismiss();
                refreshFirmwareStatus();
                rebuildGameList();
                new AlertDialog.Builder(this)
                        .setTitle("Prosperity")
                        .setMessage(r)
                        .setPositiveButton("OK", null)
                        .show();
            });
        }).start();
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_LONG).show();
    }
}
