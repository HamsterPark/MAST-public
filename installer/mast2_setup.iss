; MAST — Inno Setup script  (renamed from "MAST2"; v2.6.0+)
;
; Builds a single-file installer (MAST-Setup-v<version>-windows-x64.exe) that:
;   * installs MAST.exe + _internal/ to C:\MAST\ (default — user changeable)
;   * creates Desktop + Start Menu shortcuts
;   * on uninstall, removes ONLY the binary + _internal/, preserves all user data
;   * (Option A rename) uses a FRESH AppId; on install it DETECTS the legacy
;     v1 "MAST" and v2 "MAST2" products and offers to uninstall them first
;     (decoupled from the standalone MAST-Legacy-Uninstall.exe remover).
;
; NOTE: the file is still named mast2_setup.iss (build-internal; mast2_build.ps1
; invokes it) but it now builds the rebranded "MAST" product. The old v1
; installer mast_setup.iss is dead (kept only as the source of v1's AppId).
;
; Build:  ISCC.exe /DMAST_VERSION=2.6.0 installer\mast2_setup.iss
; Prerequisite: PyInstaller --onedir build at dist\MAST\ has run.

#ifndef MAST_VERSION
  #define MAST_VERSION "2.6.0"
#endif
#define MAST_NAME    "MAST"
#define MAST_PUBLISHER "Yuanhao Lyu"
#define MAST_URL ""
; Legacy product AppIds (for Option-A detect-and-offer-remove on install).
; NOTE the trailing `}}` — the products' `AppId={{GUID}}` registers the _is1 key
; with a literal double-close brace (verified live). Single-brace would not match.
#define V1_APPID   "{B6F2C4D7-7C4A-4E63-9F7A-MASTSTM2026}}"
#define V2_APPID   "{C8E3D6F2-9D3E-4C95-A1D9-MAST2STM2026}}"

[Setup]
; FRESH AppId for the renamed MAST line — NOT reused from v1 (B6F2) or MAST2
; (C8E3); a different codebase must not masquerade as an upgrade of either.
AppId={{D9A4E1B7-6F2C-4D85-B3E0-MAST26STM2026}}
AppName={#MAST_NAME}
AppVersion={#MAST_VERSION}
AppVerName={#MAST_NAME} v{#MAST_VERSION}
AppPublisher={#MAST_PUBLISHER}
AppPublisherURL={#MAST_URL}
AppSupportURL={#MAST_URL}
AppUpdatesURL={#MAST_URL}
DefaultDirName=C:\{#MAST_NAME}
DefaultGroupName={#MAST_NAME}
AllowNoIcons=yes
LicenseFile=
InfoBeforeFile=
InfoAfterFile=
OutputDir=..\dist
OutputBaseFilename={#MAST_NAME}-Setup-v{#MAST_VERSION}-windows-x64
SetupIconFile=..\logo\MAST2_logo.ico
; Build-speed: the bulk of the bundle is the incompressible M12 backbone
; (1.13 GB float .safetensors) + the literature index (.npy/.parquet) — those
; are stored uncompressed (see [Files] nocompression below). The remaining
; compressible payload (torch DLLs + Python) uses lzma2/normal with
; multi-threaded LZMA2 block compression.
Compression=lzma2/normal
LZMANumBlockThreads=8
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
DisableProgramGroupPage=yes
; Disable Inno's built-in close prompt: it sends WM_CLOSE, which can't reach the
; windowless `--push-server-mode` MAST.exe child (→ "can't close"). We close
; running instances ourselves in PrepareToInstall (graceful then force).
CloseApplications=no
UninstallDisplayIcon={app}\MAST.exe
UninstallDisplayName={#MAST_NAME} v{#MAST_VERSION}
DisableDirPage=no
ShowLanguageDialog=auto

[Languages]
Name: "english";  MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; 升级时先清掉**上一版的源码副本**，再让 [Files] 铺新的。
;
; Inno 只覆盖、不删除 —— 新版里没有的文件会原样留在那儿。2026-08-21 升到 6.3.0
; 之后实测：`_internal\mast\_src` 里有 787 个 .py，而这一版只该有 767，
; 多出来的 20 个全是改名前的 `campaign/`（那次改名把它整棵树换成了 `conduct/`）。
;
; 为什么单独清这一棵：`_src` 是 **eject 功能的源码来源**，也是 DP 分析子进程的
; 只读视图。留着旧副本 = 源码副本不再等于跑着的代码 —— 导出一个已经不存在的模块、
; 或者把一段死代码当成现行实现去读，而每一步看起来都正常。它还会随每次改名
; 永久累积。
;
; 只清下面点名的这两棵，不碰 `{app}` 下别的东西：`config\`、`api key\`、
; `experiments\` 都可能有操作员的数据。
Type: filesandordirs; Name: "{app}\_internal\mast\_src"

; 同一个形状的第二处：前端 bundle。Vite 的文件名带内容哈希，所以每改一次前端
; 就多出一对 `index-<hash>.js` + `.map`（约 9.5 MB），旧的**永远不会被覆盖**。
; 2026-08-21 装完 6.3.1 实测，rig 上并存着三个版本的产物：
;
;   index-BWy5p-gB.css   08-19        ← 6.2.42 一带
;   index-8rdzM7Qg.js*   08-20 21:19  ← 6.2.45
;   index-BHNm3pjG.js*   08-21 15:29  ← 6.3.1，index.html 只引用这一对
;
; 不是正确性问题（服务端读 `index.html` 决定送哪个，`/assets` 只是静态挂载，
; **不 glob 选入口**），但它每发一版涨约 9.5 MB，而且任何按 `assets/index-*.js`
; 通配去找入口的东西都会在装机树上撞到多个候选 —— 产物校验器的前端声明就是
; 那么写的。整棵 `assets\` 都是构建产物，清掉之后 [Files] 立刻铺回当前那份。
Type: filesandordirs; Name: "{app}\_internal\frontend\dist\assets"

[Files]
; Incompressible binary assets (DINOv3 backbone weights + literature index
; vectors/parquet) — store WITHOUT compression. Must precede the catch-all + be
; excluded from it so each file is packed exactly once.
Source: "..\dist\MAST\MASTv2\artifacts\*"; DestDir: "{app}\MASTv2\artifacts"; Flags: ignoreversion recursesubdirs createallsubdirs nocompression
; Everything else (MAST.exe + _internal torch/python — compressible).
Source: "..\dist\MAST\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "MASTv2\artifacts\*"

[Dirs]
Name: "{app}\api key";          Permissions: users-modify
Name: "{app}\experiments";      Permissions: users-modify
Name: "{app}\working-sessions"; Permissions: users-modify
Name: "{app}\models";           Permissions: users-modify
Name: "{app}\config\overrides"; Permissions: users-modify

[Icons]
Name: "{group}\{#MAST_NAME}";          Filename: "{app}\MAST.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MAST_NAME}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MAST_NAME}";   Filename: "{app}\MAST.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MAST_NAME}"; Filename: "{app}\MAST.exe"; WorkingDir: "{app}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\MAST.exe"; Description: "{cm:LaunchProgram,{#StringChange(MAST_NAME, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files;          Name: "{app}\MAST.exe"
Type: filesandordirs; Name: "{app}\_internal"

[Code]
var
  DataDirPage: TInputDirWizardPage;

function UninstKey(AppId: String): String;
begin
  Result := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + AppId + '_is1';
end;

// Graceful-then-force close of a running product exe (port-fragility: a
// mid-TCP force-kill can corrupt the Nanonis port until it restarts).
procedure KillExe(ExeName: String);
var rc: Integer;
begin
  Exec(ExpandConstant('{cmd}'), '/C taskkill /IM ' + ExeName, '', SW_HIDE, ewWaitUntilTerminated, rc);
  Sleep(4000);
  Exec(ExpandConstant('{cmd}'), '/C taskkill /F /T /IM ' + ExeName, '', SW_HIDE, ewWaitUntilTerminated, rc);
  Sleep(600);
end;

// Option A: offer to uninstall a legacy product (v1 MAST / v2 MAST2) before
// installing the renamed MAST. Returns silently if not present / declined.
procedure OfferRemoveLegacy(AppId, ExeName, ProdLabel: String);
var unins, ver: String; rc: Integer;
begin
  if not RegQueryStringValue(HKLM64, UninstKey(AppId), 'DisplayVersion', ver) then
    Exit;
  if MsgBox('检测到旧版 ' + ProdLabel + ' v' + ver + '。' + #13#10 +
            '本版已将 MAST2 更名为 MAST。是否现在先卸载它？' + #13#10 +
            '(仅删除程序文件，保留用户数据；也可日后用 MAST-Legacy-Uninstall.exe 清理。)',
            mbConfirmation, MB_YESNO) = IDYES then
  begin
    KillExe(ExeName);
    if RegQueryStringValue(HKLM64, UninstKey(AppId), 'UninstallString', unins) then
      Exec(RemoveQuotes(unins), '/SILENT /SUPPRESSMSGBOXES /NORESTART', '', SW_SHOW, ewWaitUntilTerminated, rc);
  end;
end;

function InitializeSetup(): Boolean;
var
  PriorVersion: String;
  Confirmed: Integer;
begin
  Result := True;
  // In-place upgrade of MAST itself (same fresh AppId). The registered _is1 key
  // carries a trailing `}}` (AppId={{...}}), so match that here too.
  if RegQueryStringValue(HKLM64, UninstKey('{D9A4E1B7-6F2C-4D85-B3E0-MAST26STM2026}}'), 'DisplayVersion', PriorVersion) then
  begin
    Confirmed := MsgBox(
      '检测到已安装的 MAST v' + PriorVersion + #13#10 +
      '本次升级到 v{#MAST_VERSION} 将覆盖 MAST.exe 与 _internal\，但保留所有用户数据。' + #13#10#13#10 +
      '继续安装？',
      mbConfirmation, MB_YESNO);
    if Confirmed = IDNO then
    begin
      Result := False;
      Exit;
    end;
  end;
  // Legacy products (the rename's whole point): offer to clean them up.
  OfferRemoveLegacy('{#V1_APPID}', 'MAST.exe', 'MAST v1');
  OfferRemoveLegacy('{#V2_APPID}', 'MAST2.exe', 'MAST2');
end;

procedure InitializeWizard;
begin
  DataDirPage := CreateInputDirPage(
    wpSelectDir,
    '数据保存目录',
    '所有实验数据、对话历史、API key、日志等都会保存在这里',
    '请选择 MAST 的"用户数据目录"。' + #13#10#13#10 +
    '安装后产生的所有数据（实验记录、对话、API key、TTS 缓存、Nanonis 会话、' +
    '训练好的模型、配置 override、启动器/服务日志）都保存在此目录下。' + #13#10#13#10 +
    '升级或卸载 MAST 时，此目录的内容**不会**被删除或覆盖。',
    False, '');
  DataDirPage.Add('数据目录:');
  DataDirPage.Values[0] := ExpandConstant('{sd}\MAST-data');
end;

procedure CurPageChanged(CurPageID: Integer);
var
  PriorMarker: String;
  PriorMarkerLines: TArrayOfString;
  ExistingPath: String;
begin
  if (DataDirPage <> nil) and (CurPageID = DataDirPage.ID) then
  begin
    PriorMarker := ExpandConstant('{app}\data_dir.txt');
    if FileExists(PriorMarker) then
    begin
      if LoadStringsFromFile(PriorMarker, PriorMarkerLines) and
         (GetArrayLength(PriorMarkerLines) >= 1) then
      begin
        ExistingPath := Trim(PriorMarkerLines[0]);
        if (ExistingPath <> '') and DirExists(ExistingPath) then
          DataDirPage.Values[0] := ExistingPath;
      end;
    end
    else if DirExists(ExpandConstant('{app}\experiments')) or
            DirExists(ExpandConstant('{app}\api key')) then
    begin
      DataDirPage.Values[0] := ExpandConstant('{app}');
    end;
  end;
end;

procedure SHChangeNotify(wEventId: Cardinal; uFlags: Cardinal; dwItem1, dwItem2: Cardinal);
  external 'SHChangeNotify@shell32.dll stdcall';

procedure NotifyIconCacheChanged;
begin
  try
    SHChangeNotify($08000000, 0, 0, 0);
  except
  end;
end;

function IsMASTRunning(): Boolean;
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{cmd}'),
       '/C tasklist /FI "IMAGENAME eq MAST.exe" /NH | find /I "MAST.exe"',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := (ResultCode = 0);
end;

// KNOWN_ISSUES §3.1 — the factory literature index ships in the package and the
// [Files] entry above copies it over {app}\MASTv2\artifacts\literature_index\
// with `ignoreversion`. That is the SAME directory the agent's own fetched
// papers are appended into (ingest._promote_to_big writes metadata.parquet +
// vectors.npy there), and nothing merges. So an upgrade used to delete every
// paper the operator's rig had collected, silently. The documented mitigation
// was "remember to back it up first", which is not a mitigation.
//
// Snapshot it here — BEFORE any file is overwritten. A COPY, not a move: the
// machine must never be in a state where it has no index, not even for the
// seconds between this and the file copy. mast.knowledge.index_merge picks the
// snapshot up on the next launch, merges back the rows the new index lacks, and
// then DELETES snapshots that turned out to add nothing (those are redundant by
// construction, so deleting them cannot lose information) — that is what keeps
// 205 MB from accumulating per upgrade.
procedure SnapshotLiteratureIndex();
var
  IndexDir, SnapDir, Stamp: String;
  rc: Integer;
begin
  IndexDir := ExpandConstant('{app}\MASTv2\artifacts\literature_index');
  // metadata.parquet, not the directory: an empty dir is nothing to protect.
  if not FileExists(IndexDir + '\metadata.parquet') then
    Exit;
  Stamp := GetDateTimeString('yyyymmdd-hhnnss', '-', '');
  SnapDir := ExpandConstant('{app}\MASTv2\artifacts\literature_index.pre-upgrade-') + Stamp;
  // robocopy exit codes 0..7 are success; 8+ are real failures. Exec's own
  // Boolean result must be checked too — if the shell never launched, rc is
  // never assigned and testing it alone would report a phantom success.
  rc := 16;
  if not Exec(ExpandConstant('{cmd}'),
       '/C robocopy "' + IndexDir + '" "' + SnapDir + '" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP',
       '', SW_HIDE, ewWaitUntilTerminated, rc) then
    rc := 16;
  if rc >= 8 then
    MsgBox('无法备份文献索引（robocopy 返回 ' + IntToStr(rc) + '）。' + #13#10 +
           '继续安装会覆盖 ' + IndexDir + '，' + #13#10 +
           '其中包含本机自主取文攒下的文献。' + #13#10#13#10 +
           '建议先手动复制该目录再继续。',
           mbError, MB_OK);
end;

// Run BEFORE overwriting files (can abort the install). Prompt first.
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if IsMASTRunning() then
  begin
    if MsgBox(
         '检测到 MAST 正在运行（可能包括无窗口的后台更新服务器）。' + #13#10 +
         '安装需要先关闭它才能覆盖程序文件。' + #13#10#13#10 +
         '请先在 MAST 中保存未完成的工作、停止正在进行的扫描。' + #13#10#13#10 +
         '点"是"= 关闭 MAST 并继续安装；点"否"= 取消安装。',
         mbConfirmation, MB_YESNO) = IDYES then
    begin
      KillExe('MAST.exe');
      if IsMASTRunning() then
      begin
        Result := '仍检测到 MAST 在运行，无法继续安装。' + #13#10 +
                  '请在任务管理器中手动结束所有 MAST.exe 后，重新运行安装程序。';
        Exit;
      end;
    end
    else
    begin
      Result := '安装已取消：请先关闭 MAST，再运行安装程序。';
      Exit;
    end;
  end;
  // Past this point the install WILL overwrite files, and MAST is confirmed
  // stopped — so the index is not mid-write. Both conditions matter, which is
  // why the snapshot sits here rather than at the top.
  SnapshotLiteratureIndex();
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ChosenDir: String;
  Sub: String;
  Subs: array[0..5] of String;
  i: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    ChosenDir := DataDirPage.Values[0];
    if Trim(ChosenDir) = '' then
      ChosenDir := ExpandConstant('{app}');

    ForceDirectories(ChosenDir);

    Subs[0] := 'experiments';
    Subs[1] := 'experiments\logs';
    Subs[2] := 'api key';
    Subs[3] := 'working-sessions';
    Subs[4] := 'models';
    Subs[5] := 'config\overrides';
    for i := 0 to 5 do
    begin
      Sub := ChosenDir + '\' + Subs[i];
      ForceDirectories(Sub);
    end;

    SaveStringToFile(
      ExpandConstant('{app}\data_dir.txt'),
      ChosenDir + #13#10,
      False);

    NotifyIconCacheChanged;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDirContent: TArrayOfString;
  DataDirPath: String;
  MarkerPath: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDirPath := ExpandConstant('{app}');
    MarkerPath := ExpandConstant('{app}\data_dir.txt');
    if FileExists(MarkerPath) then
    begin
      if LoadStringsFromFile(MarkerPath, DataDirContent) and
         (GetArrayLength(DataDirContent) >= 1) then
      begin
        if Trim(DataDirContent[0]) <> '' then
          DataDirPath := Trim(DataDirContent[0]);
      end;
    end;

    MsgBox(
      'MAST 已卸载（仅删除程序文件）。' + #13#10 +
      '以下用户数据被保留在 ' + DataDirPath + '：' + #13#10 +
      '  - experiments\（实验记录、对话、TTS 缓存、计划、日志）' + #13#10 +
      '  - api key\（LLM API key + LAN auth）' + #13#10 +
      '  - working-sessions\（Nanonis 扫描会话）' + #13#10 +
      '  - models\（训练好的 ML 模型）' + #13#10 +
      '  - config\（admin override）' + #13#10#13#10 +
      '如需彻底清除，请手动删除上述目录。',
      mbInformation, MB_OK);
  end;
end;
