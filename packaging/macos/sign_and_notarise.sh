#!/bin/sh
# Sign, notarise and staple DexVantage.app (Apple silicon) for distribution outside the App Store. NOT run by the build
# or CI: it needs the maintainer's Apple Developer ID certificate in the login keychain and a notarytool profile.
#
#   packaging/macos/sign_and_notarise.sh out/dist/DexVantage.app "Developer ID Application: NAME (TEAMID)" PROFILE
#
# PROFILE is a keychain profile made once with
#   xcrun notarytool store-credentials PROFILE --apple-id APPLE_ID --team-id TEAMID
# (it asks for an app-specific password; nothing secret goes in this repository). Steps, as Apple documents them
# ("Notarizing macOS software before distribution", developer.apple.com):
#   1. sign every Mach-O file inside the app (libraries and extension modules first, the launchers, then the app)
#      with the hardened runtime and a secure timestamp;
#   2. zip the app with ditto and submit it to the notary service, waiting for the verdict;
#   3. staple the ticket to the app and zip it again for the release (the stapled zip replaces the unsigned one).
set -eu
APP=$1
IDENTITY=$2
PROFILE=$3
HERE=$(cd "$(dirname "$0")" && pwd)
ENT="$HERE/entitlements.plist"

# 1. every Mach-O file, deepest first, then the launchers and the app itself
find "$APP/Contents" -type f \( -name '*.dylib' -o -name '*.so' -o -perm -u+x \) -print0 |
    xargs -0 file | grep 'Mach-O' | cut -d: -f1 | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2- |
    while IFS= read -r f; do
        codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$IDENTITY" "$f"
    done
codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

# 2. notarise
ZIP="${APP%.app}-notarise.zip"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait
rm -f "$ZIP"

# 3. staple and check what Gatekeeper says
xcrun stapler staple "$APP"
spctl --assess --type execute --verbose=2 "$APP"
echo "Signed and notarised: $APP. Zip it again with: ditto -c -k --sequesterRsrc --keepParent \"$APP\" OUT.zip"
