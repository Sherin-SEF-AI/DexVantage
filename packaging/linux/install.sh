#!/bin/sh
# Install DexVantage for the current user (no root needed):
#   ~/.local/opt/dexvantage         the program
#   ~/.local/bin/dexvantage         command for batch mode, --self-test and starting the window from a terminal
#   menu entry, icon and the .dexv file type in $XDG_DATA_HOME (default ~/.local/share)
# Run it from the extracted folder:  ./install.sh
# PREFIX=/some/where ./install.sh installs into /some/where/opt/dexvantage, /some/where/bin and /some/where/share.
# Ported from Vantage packaging/linux/install.sh.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${PREFIX:-}" ]; then
    DATA="$PREFIX/share"
else
    PREFIX="$HOME/.local"
    DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
fi
DEST="$PREFIX/opt/dexvantage"
MARK=.dexvantage-install

if [ ! -x "$HERE/DexVantage" ] || [ ! -x "$HERE/dexvantage-cli" ]; then
    echo "install.sh: run me from the extracted DexVantage folder (DexVantage and dexvantage-cli must be next to me)" >&2
    exit 1
fi
if [ -e "$DEST" ] && [ ! -e "$DEST/$MARK" ]; then
    echo "install.sh: $DEST exists and was not made by this installer; move it away first" >&2
    exit 1
fi
if [ "$HERE" = "$DEST" ]; then
    echo "install.sh: already installed in $DEST" >&2
    exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST" "$PREFIX/bin" "$DATA/applications" "$DATA/icons/hicolor/256x256/apps" "$DATA/mime/packages"
cp -R "$HERE/." "$DEST/"
printf 'data=%s\nbin=%s\n' "$DATA" "$PREFIX/bin" > "$DEST/$MARK"     # where uninstall.sh finds what this wrote
ln -sf "$DEST/dexvantage-cli" "$PREFIX/bin/dexvantage"
cp "$DEST/_internal/dexvantage/resources/icons/dexvantage.png" "$DATA/icons/hicolor/256x256/apps/dexvantage.png"

# the Exec line, as the Desktop Entry spec wants it: the path in double quotes with " ` $ \ escaped by a backslash,
# that backslash (and a literal one) escaped again because a desktop file's values are unescaped before the quotes
# are read (GLib rejects the entry otherwise), and % doubled
exe=$(printf '%s' "$DEST/DexVantage" | sed -e 's/\\/\\\\\\\\/g' -e 's/["`$]/\\\\&/g' -e 's/%/%%/g')
while IFS= read -r line; do
    case "$line" in
        Exec=*) printf 'Exec="%s" %%f\n' "$exe" ;;
        *) printf '%s\n' "$line" ;;
    esac
done < "$HERE/dexvantage.desktop" > "$DATA/applications/dexvantage.desktop"
cp "$HERE/dexvantage-mime.xml" "$DATA/mime/packages/dexvantage.xml"

command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$DATA/mime" >/dev/null 2>&1 || true
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true

echo "DexVantage installed in $DEST"
echo "  menu:      DexVantage (under Science / Engineering)"
echo "  terminal:  $PREFIX/bin/dexvantage   (dexvantage --self-test checks the installation)"
echo "  uninstall: $DEST/uninstall.sh"
case ":$PATH:" in
    *":$PREFIX/bin:"*) ;;
    *) echo "  note: $PREFIX/bin is not on your PATH" ;;
esac
