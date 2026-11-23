' Sonika Storyteller launcher: starts the local voice server (hidden) and opens the app window.
' Set the environment variable SONIKA_PYTHON to the python.exe of the environment where Chatterbox is installed.
' If it is not set, the first "python" on your PATH is used.
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
py = sh.ExpandEnvironmentStrings("%SONIKA_PYTHON%")
If py = "%SONIKA_PYTHON%" Or py = "" Then py = "python"
url = "http://127.0.0.1:7861"

Function IsUp()
  On Error Resume Next
  Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
  http.setTimeouts 1000, 1000, 1000, 1000
  http.Open "GET", url & "/api/voices", False
  http.Send
  IsUp = (Err.Number = 0 And http.Status = 200)
End Function

If Not IsUp() Then
  sh.CurrentDirectory = appDir
  sh.Run "cmd /c """"" & py & """ """ & appDir & "\server.py"" > """ & appDir & "\data\server_out.log"" 2>&1""", 0, False
  For i = 1 To 60
    WScript.Sleep 500
    If IsUp() Then Exit For
  Next
End If

chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
If fso.FileExists(chrome) Then
  sh.Run """" & chrome & """ --app=" & url & " --window-size=1320,880", 1, False
Else
  sh.Run url, 1, False
End If
