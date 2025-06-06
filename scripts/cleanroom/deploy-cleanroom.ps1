param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$location = "westeurope",
    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$artefactsDir = "$privateDir/$contractId-artefacts",

    [string]$cleanRoomName = "cleanroom-$contractId",
    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$cleanroomEndpoint = "$publicDir/$cleanRoomName.endpoint"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Test-AzureAccessToken

Write-Log OperationStarted `
    "Generating CA cert for contract '$contractId'..." 
az cleanroom governance ca generate-key `
    --contract-id $contractId `
    --governance-client $cgsClient
# Get the CA cert generated for the deployment.
az cleanroom governance ca show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "caCert" `
    --output tsv > $artefactsDir/$cleanRoomName-ca.crt
Write-Log OperationCompleted `
    "Generated CA cert for contract '$contractId': '$artefactsDir/$cleanRoomName-ca.crt'." 

Write-Log OperationStarted `
    "Deploying clean room for contract '$contractId'..." 
# Get the agreed upon ARM template for deployment.
(az cleanroom governance deployment template show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "data") | Out-File "$artefactsDir/accepted-deployment-template.json"

# Deploy the clean room.
az deployment group create `
    --resource-group $resourceGroup `
    --name $cleanRoomName `
    --template-file "$artefactsDir/accepted-deployment-template.json" `
    --parameters location=$location

Write-Log OperationCompleted `
    "Deployed clean room '$cleanRoomName' for contract '$contractId' to '$resourceGroup'." 

while ($true) {
    $ccrIP = az container show `
        --name $cleanRoomName `
        -g $resourceGroup `
        --query "ipAddress.ip" `
        --output tsv
    if ($null -eq $ccrIP) {
        Write-Log Information `
            "$(Get-TimeStamp) Clean room IP for '$cleanRoomName' is not yet available. Waiting for 20 seconds..."
        Start-Sleep -Seconds 20
    }
    else {
        break
    }
}

Write-Host "Clean Room IP address: $ccrIP"

$ccrIP | Out-File "$cleanroomEndpoint"
Write-Log OperationCompleted `
    "CCR endpoint details {IP: '$ccrIp'} written to '$cleanroomEndpoint'."