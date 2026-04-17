param(
    [string]$ApiKey,
    [string]$Model = "gemini-2.5-flash"
)

if (-not $ApiKey) {
    $ApiKey = Read-Host "Enter your Gemini API key"
}

if (-not $ApiKey) {
    throw "GEMINI_API_KEY was not provided."
}

[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $ApiKey, "User")
[Environment]::SetEnvironmentVariable("GEMINI_MODEL", $Model, "User")
[Environment]::SetEnvironmentVariable("LLM_PROVIDER", "gemini", "User")

Write-Host "Saved GEMINI_API_KEY, GEMINI_MODEL, and LLM_PROVIDER to the current user's environment."
