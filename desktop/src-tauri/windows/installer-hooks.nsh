; Photo Select keeps unsaved reviews safe: never terminate the app or worker.
; Tauri includes utils.nsh before this file, so replace its process-check macro.
!macroundef CheckIfAppIsRunning

Var PhotoSelectCheckSession
Var PhotoSelectCheckResult
Var PhotoSelectUnsafeResources

; NSIS RMDir /r follows directory junctions. Refuse replacement if the bundled
; resource tree contains any reparse point, so cleanup cannot reach other data.
; Takes a path on the stack and restores its temporary registers.
Function PhotoSelectCheckResourceTree
  Exch $0
  Push $1
  Push $2
  Push $3
  Push $4
  Push $5
  Push $6
  System::Call 'kernel32::GetFileAttributesW(w r0) i .r1 ?e'
  Pop $6 ; Capture the API error before any other calls can overwrite it.
  ${If} $1 = -1
    ; A missing resource directory is normal for a fresh install.
    ${If} $6 <> 2
    ${AndIf} $6 <> 3
      StrCpy $PhotoSelectUnsafeResources 1
    ${EndIf}
    Goto photo_select_tree_done
  ${EndIf}
  IntOp $3 $1 & 0x400 ; FILE_ATTRIBUTE_REPARSE_POINT
  ${If} $3 <> 0
    StrCpy $PhotoSelectUnsafeResources 1
    Goto photo_select_tree_done
  ${EndIf}
  IntOp $1 $1 & 0x10 ; FILE_ATTRIBUTE_DIRECTORY
  ${If} $1 <> 0
    ; WIN32_FIND_DATAW: 44 bytes before cFileName[260], then 14 WCHARs.
    ; 44 + 520 + 28 = 592 bytes on both Windows architectures.
    ; SDK layout: microsoft/win32metadata generation/WinSDK/RecompiledIdlHeaders/um/minwinbase.h
    System::Alloc 592
    Pop $4
    ${If} $4 = 0
      StrCpy $PhotoSelectUnsafeResources 1
      Goto photo_select_tree_done
    ${EndIf}
    System::Call 'kernel32::FindFirstFileW(w "$0\*.*", p r4) p .r2 ?e'
    Pop $6
    ${If} $2 = -1
      ; Empty directories are valid; other failures block all cleanup.
      ${If} $6 <> 2
      ${AndIf} $6 <> 18
        StrCpy $PhotoSelectUnsafeResources 1
      ${EndIf}
    ${Else}
      photo_select_tree_next:
        System::Call '*$4(&v44, &w260 .r3)'
        ; LogicLib != compares strings; <> converts filenames to integers.
        ${If} $3 != "."
        ${AndIf} $3 != ".."
          Push "$0\$3"
          Call PhotoSelectCheckResourceTree
        ${EndIf}
        ${If} $PhotoSelectUnsafeResources <> 1
          System::Call 'kernel32::FindNextFileW(p r2, p r4) i .r5 ?e'
          Pop $6
          ${If} $5 = 0
            ${If} $6 <> 18 ; ERROR_NO_MORE_FILES is the only valid end.
              StrCpy $PhotoSelectUnsafeResources 1
            ${EndIf}
            Goto photo_select_tree_close
          ${EndIf}
          Goto photo_select_tree_next
        ${EndIf}
      photo_select_tree_close:
        System::Call 'kernel32::FindClose(p r2)'
    ${EndIf}
    System::Free $4
  ${EndIf}
  photo_select_tree_done:
    Pop $6
    Pop $5
    Pop $4
    Pop $3
    Pop $2
    Pop $1
    Pop $0
FunctionEnd

!macro CheckIfAppIsRunning executablePath productName
  !define PhotoSelectCheckID ${__LINE__}

  photo_select_check_${PhotoSelectCheckID}:
    !insertmacro RestartManager_StartSession $PhotoSelectCheckSession
    ${If} $PhotoSelectCheckSession == ""
      Goto photo_select_check_failed_${PhotoSelectCheckID}
    ${EndIf}

    ; The worker can outlive the window; check both executables before changing
    ; files. Restart Manager also sees processes holding either executable.
    ${If} ${FileExists} "${executablePath}"
      !insertmacro RestartManager_RegisterFile $PhotoSelectCheckSession "${executablePath}"
      ${If} $0 <> 0
        !insertmacro RestartManager_EndSession $PhotoSelectCheckSession
        Goto photo_select_check_failed_${PhotoSelectCheckID}
      ${EndIf}
    ${EndIf}
    ${If} ${FileExists} "$INSTDIR\resources\engine\photo-select-engine.exe"
      !insertmacro RestartManager_RegisterFile $PhotoSelectCheckSession "$INSTDIR\resources\engine\photo-select-engine.exe"
      ${If} $0 <> 0
        !insertmacro RestartManager_EndSession $PhotoSelectCheckSession
        Goto photo_select_check_failed_${PhotoSelectCheckID}
      ${EndIf}
    ${EndIf}

    StrCpy $1 0
    StrCpy $2 0
    StrCpy $3 0
    System::Call 'RSTRTMGR::RmGetList(i $PhotoSelectCheckSession, *i .r1, *i .r2, p 0, *i .r3) i .r0'
    StrCpy $PhotoSelectCheckResult $0
    !insertmacro RestartManager_EndSession $PhotoSelectCheckSession

    ${If} $PhotoSelectCheckResult = 0
      Goto photo_select_check_done_${PhotoSelectCheckID}
    ${ElseIf} $PhotoSelectCheckResult = ${ERROR_MORE_DATA}
      DetailPrint "${productName} or its analysis worker is running. Close it before continuing."
      IfSilent photo_select_check_cancel_${PhotoSelectCheckID}
      ${IfThen} $PassiveMode = 1 ${|} Goto photo_select_check_cancel_${PhotoSelectCheckID} ${|}
      MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "Save your review and close ${productName}. Wait for analysis to stop, then click Retry. Setup will leave the app and its worker running until you close them." IDRETRY photo_select_check_${PhotoSelectCheckID} IDCANCEL photo_select_check_cancel_${PhotoSelectCheckID}
    ${EndIf}

  photo_select_check_failed_${PhotoSelectCheckID}:
    DetailPrint "Setup could not check whether ${productName} is running. No application files were changed."
    IfSilent photo_select_check_cancel_${PhotoSelectCheckID}
    ${IfThen} $PassiveMode = 1 ${|} Goto photo_select_check_cancel_${PhotoSelectCheckID} ${|}
    MessageBox MB_OK|MB_ICONSTOP "Setup could not check whether ${productName} is running. Close the app and its analysis worker, then run Setup again."

  photo_select_check_cancel_${PhotoSelectCheckID}:
    SetErrorLevel 2
    Quit

  photo_select_check_done_${PhotoSelectCheckID}:
  !undef PhotoSelectCheckID
!macroend

!macro NSIS_HOOK_PREINSTALL
  ; Only bundled, replaceable assets live here. Projects/cache/settings live
  ; separately under $LOCALAPPDATA\com.shurkantwo.photoselect and are untouched.
  ; The custom template runs this hook AFTER checking the app and worker.
  StrCpy $PhotoSelectUnsafeResources 0
  Push "$INSTDIR\resources"
  Call PhotoSelectCheckResourceTree
  ${If} $PhotoSelectUnsafeResources = 1
    DetailPrint "Bundled resources contain a filesystem link or cannot be inspected. Setup did not remove them."
    IfSilent photo_select_unsafe_resources_cancel
    ${IfThen} $PassiveMode = 1 ${|} Goto photo_select_unsafe_resources_cancel ${|}
    MessageBox MB_OK|MB_ICONSTOP "Setup cannot safely replace the bundled resources because they contain a filesystem link or cannot be inspected. Restore the original resource folders and run Setup again."
    photo_select_unsafe_resources_cancel:
      SetErrorLevel 3
      Quit
  ${EndIf}
  ClearErrors
  ${If} ${FileExists} "$INSTDIR\resources\engine\*.*"
    RMDir /r "$INSTDIR\resources\engine"
  ${EndIf}
  ${If} ${FileExists} "$INSTDIR\resources\lightroom\*.*"
    RMDir /r "$INSTDIR\resources\lightroom"
  ${EndIf}
  ${If} ${Errors}
    DetailPrint "Cannot replace bundled engine/Lightroom files. Close programs using them and retry Setup."
    IfSilent photo_select_cleanup_cancel
    ${IfThen} $PassiveMode = 1 ${|} Goto photo_select_cleanup_cancel ${|}
    MessageBox MB_OK|MB_ICONSTOP "Cannot replace the bundled analysis engine or Lightroom plugin. Close programs using those files, then run Setup again. Your projects and original photos are unchanged."
    photo_select_cleanup_cancel:
      SetErrorLevel 3
      Quit
  ${EndIf}
!macroend
