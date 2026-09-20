; MAST 旧版卸载工具 — standalone remover for legacy MAST (v1) + MAST2 (v2)
; ============================================================================
; Why: starting with the renamed "MAST" (2.6.0+), updates are incremental (OTA
; delta), so the full installer is NOT re-run every release — i.e. installing a
; new version no longer implies uninstalling the old one. This standalone tool
; lets a user fully remove the OLD products (v1 "MAST" at C:\MAST and v2 "MAST2"
; at C:\MAST2) on demand, independent of any installer.
;
; It is NOT an installer: installs no files, registers no Add/Remove entry. Per
; user checkbox it runs each detected product's registered uninstaller silently
; (direct-delete fallback), always preserving the user-data directory.
;
; DETECTION (fixed 2026-06-09 after the first build found nothing):
;   * The products install in 64-bit mode, so their Inno _is1 uninstall keys live
;     in the NATIVE 64-bit registry view — we MUST read HKLM64 (this tool now also
;     runs 64-bit via ArchitecturesInstallIn64BitMode). A 32-bit read hits the
;     empty WOW6432Node and finds nothing.
;   * The products' AppId source was `{{GUID}}` (double-close), so the REGISTERED
;     key name carries a trailing DOUBLE brace: `{<GUID>}}_is1`. The exact keys
;     (verified in the live registry) are hard-coded below. v1 and the renamed
;     MAST 2.6 share C:\MAST, so detection MUST be by AppId (registry), not dir.
;
; Build:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\mast_legacy_uninstall.iss
; Output: dist\MAST-Legacy-Uninstall.exe

#define TOOL_NAME    "MAST 旧版卸载工具"
#define TOOL_VERSION "1.0.0"

[Setup]
AppId={{E7A1F0C3-4B5D-46E8-9A2F-MASTUNINST2026}}
AppName={#TOOL_NAME}
AppVersion={#TOOL_VERSION}
AppVerName={#TOOL_NAME} v{#TOOL_VERSION}
AppPublisher=Yuanhao Lyu
OutputDir=..\dist
OutputBaseFilename=MAST-Legacy-Uninstall
DefaultDirName={tmp}\mast-legacy-uninstall
DisableDirPage=yes
DisableProgramGroupPage=yes
Uninstallable=no
CreateUninstallRegKey=no
CreateAppDir=no
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
; Run 64-bit so the default registry view + HKLM64 see the products' native-view
; _is1 uninstall keys (the whole reason the first build detected nothing).
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
ShowLanguageDialog=no
SetupIconFile=..\logo\MAST2_logo.ico

[Languages]
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Code]
const
  // EXACT registered uninstall keys (verified live). Note the trailing `}}`
  // (the products' AppId was `{{GUID}}`). Read from the 64-bit view (HKLM64).
  V1_KEY = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{B6F2C4D7-7C4A-4E63-9F7A-MASTSTM2026}}_is1';
  V2_KEY = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{C8E3D6F2-9D3E-4C95-A1D9-MAST2STM2026}}_is1';

var
  ChoicePage: TInputOptionWizardPage;
  V1Dir, V1Unins, V1Ver: String;
  V2Dir, V2Unins, V2Ver: String;
  V1Present, V2Present: Boolean;
  V1Idx, V2Idx: Integer;

// Read a product's uninstall info from the 64-bit registry view. Returns True if
// the product is installed (its _is1 key exists).
function Detect(Key: String; var Dir, Unins, Ver: String): Boolean;
var us: String;
begin
  Dir := ''; Unins := ''; Ver := '';
  Result := RegQueryStringValue(HKLM64, Key, 'UninstallString', us);
  if Result then
  begin
    Unins := RemoveQuotes(Trim(us));
    RegQueryStringValue(HKLM64, Key, 'InstallLocation', Dir);
    Dir := RemoveBackslash(Trim(Dir));
    RegQueryStringValue(HKLM64, Key, 'DisplayVersion', Ver);
  end;
end;

// Graceful-then-force close of a running product exe (port-fragility: a mid-TCP
// force-kill can corrupt the Nanonis port until it restarts).
procedure KillExe(ExeName: String);
var rc: Integer;
begin
  Exec(ExpandConstant('{cmd}'), '/C taskkill /IM ' + ExeName, '', SW_HIDE, ewWaitUntilTerminated, rc);
  Sleep(4000);
  Exec(ExpandConstant('{cmd}'), '/C taskkill /F /T /IM ' + ExeName, '', SW_HIDE, ewWaitUntilTerminated, rc);
  Sleep(600);
end;

procedure RemoveOne(Dir, Unins, ExeName, GroupName: String);
var rc: Integer;
begin
  KillExe(ExeName);
  if (Unins <> '') and FileExists(Unins) then
    // Preferred: the product's own uninstaller (removes binary + _internal,
    // preserves user data).
    Exec(Unins, '/SILENT /SUPPRESSMSGBOXES /NORESTART', '', SW_SHOW, ewWaitUntilTerminated, rc)
  else if Dir <> '' then
  begin
    // Fallback (registered uninstaller gone): delete ONLY program files.
    DelTree(Dir + '\_internal', True, True, True);
    DeleteFile(Dir + '\' + ExeName);
    DelTree(ExpandConstant('{commonprograms}\' + GroupName), True, True, True);
  end;
end;

function CapFor(Present: Boolean; Name, Ver, Dir: String): String;
begin
  if Present then
    Result := Name + '  (v' + Ver + ',  ' + Dir + ')'
  else
    Result := Name + '  (未安装)';
end;

procedure InitializeWizard;
begin
  // Detect BEFORE adding the items so each row is added with its final caption
  // (TNewCheckListBox has no reliable settable ItemCaption post-Add).
  V1Present := Detect(V1_KEY, V1Dir, V1Unins, V1Ver);
  V2Present := Detect(V2_KEY, V2Dir, V2Unins, V2Ver);

  ChoicePage := CreateInputOptionPage(
    wpWelcome,
    '选择要卸载的旧版本',
    '本工具仅删除程序文件,保留你的用户数据(实验记录 / api key / 会话 / 模型 / 配置)。',
    '检测到以下已安装的旧版 MAST。勾选要卸载的项,然后点"下一步"。' + #13#10 +
    '(运行中的实例会被先优雅关闭,再卸载。)',
    True, False);
  V1Idx := ChoicePage.Add(CapFor(V1Present, 'MAST v1', V1Ver, V1Dir));
  V2Idx := ChoicePage.Add(CapFor(V2Present, 'MAST2', V2Ver, V2Dir));

  ChoicePage.CheckListBox.ItemEnabled[V1Idx] := V1Present;
  ChoicePage.CheckListBox.ItemEnabled[V2Idx] := V2Present;
  ChoicePage.Values[V1Idx] := V1Present;
  ChoicePage.Values[V2Idx] := V2Present;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (ChoicePage <> nil) and (CurPageID = ChoicePage.ID) then
    if (not V1Present) and (not V2Present) then
      MsgBox('未检测到已安装的 MAST v1 或 MAST2(已查 64 位注册表)。无需卸载。',
             mbInformation, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  removed: Integer;
begin
  if CurStep = ssInstall then
  begin
    removed := 0;
    if V1Present and ChoicePage.Values[V1Idx] then
    begin
      RemoveOne(V1Dir, V1Unins, 'MAST.exe', 'MAST');
      removed := removed + 1;
    end;
    if V2Present and ChoicePage.Values[V2Idx] then
    begin
      RemoveOne(V2Dir, V2Unins, 'MAST2.exe', 'MAST2');
      removed := removed + 1;
    end;
    if removed > 0 then
      MsgBox('完成。已卸载所选旧版(程序文件已删除,用户数据保留)。' + #13#10 +
             '如需彻底清除数据,请手动删除对应的数据目录(如 C:\MAST2-data)。',
             mbInformation, MB_OK);
  end;
end;
