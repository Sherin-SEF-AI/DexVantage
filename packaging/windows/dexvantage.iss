; Inno Setup script for the Windows installer. packaging/build.py runs it after PyInstaller:
;   iscc /DAppVersion=0.1.0 /DBundleDir=out\dist\DexVantage /Oout\dist packaging\windows\dexvantage.iss
; Installs per user with no administrator rights (PrivilegesRequired=lowest: into %LOCALAPPDATA%\Programs\DexVantage,
; the Start menu entry and the .dexv file type under HKCU); the setup offers an all-users install too, which then asks
; for administrator rights. Relative paths here are relative to this file's folder.
; Ported from Vantage packaging/windows/vantage.iss.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef BundleDir
  #define BundleDir "..\..\out\dist\DexVantage"
#endif

[Setup]
AppId={{0447E9FB-A45D-48B0-9D27-D5C0F2EFEA48}
AppName=DexVantage
AppVersion={#AppVersion}
AppVerName=DexVantage {#AppVersion}
AppComments=Camera, depth and teleop planning for robot-learning data collection
AppPublisher=DexVantage
AppPublisherURL=https://github.com/Sherin-SEF-AI/DexVantage
AppSupportURL=https://github.com/Sherin-SEF-AI/DexVantage/issues
AppUpdatesURL=https://github.com/Sherin-SEF-AI/DexVantage/releases
VersionInfoVersion={#AppVersion}
VersionInfoProductName=DexVantage
DefaultDirName={autopf}\DexVantage
DefaultGroupName=DexVantage
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=DexVantage-{#AppVersion}-windows-x64-setup
SetupIconFile=..\icons\dexvantage.ico
LicenseFile=..\..\LICENSE
UninstallDisplayIcon={app}\DexVantage.exe
UninstallDisplayName=DexVantage {#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc"; Description: "Open .dexv projects with DexVantage"; GroupDescription: "File types:"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; an update replaces the program folder's contents: files an older version had and this one does not go
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\DexVantage"; Filename: "{app}\DexVantage.exe"
; cmd /k drops the first and the last quote of its command line unless exactly two quotes enclose a path with a space
; in it; the extra outer pair keeps the program's path quoted wherever it was installed
Name: "{group}\DexVantage self-test"; Filename: "{cmd}"; Parameters: "/k """"{app}\dexvantage-cli.exe"" --self-test"""; IconFilename: "{app}\DexVantage.exe"
Name: "{autodesktop}\DexVantage"; Filename: "{app}\DexVantage.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.dexv\OpenWithProgids"; ValueType: string; ValueName: "DexVantage.Project"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\DexVantage.Project"; ValueType: string; ValueName: ""; ValueData: "DexVantage project"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\DexVantage.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\DexVantage.exe,0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\DexVantage.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\DexVantage.exe"" ""%1"""; Tasks: assoc

[Run]
Filename: "{app}\DexVantage.exe"; Description: "{cm:LaunchProgram,DexVantage}"; Flags: nowait postinstall skipifsilent
