$ErrorActionPreference = "Stop"

$secure = Read-Host "Enter DeepSeek API key (input is hidden)" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if ([string]::IsNullOrWhiteSpace($plain)) {
        throw "API key cannot be empty"
    }
    [Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", $plain, "User")
    $env:DEEPSEEK_API_KEY = $plain
    Write-Host "DEEPSEEK_API_KEY saved. Close and reopen VS Code."
}
finally {
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    $plain = $null
    $secure.Dispose()
}
