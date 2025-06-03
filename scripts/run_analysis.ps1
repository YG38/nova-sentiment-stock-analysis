# Create output directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "../output/eda_plots" | Out-Null

# Run the EDA script
Write-Host "Starting EDA analysis..." -ForegroundColor Green
python task1_eda.py

# Check if plots were generated
$plotCount = (Get-ChildItem "../output/eda_plots" -File).Count
if ($plotCount -gt 0) {
    Write-Host "\n✅ Analysis complete! Generated $plotCount plot(s) in ../output/eda_plots/" -ForegroundColor Green
} else {
    Write-Host "\n❌ No plots were generated. Please check for errors." -ForegroundColor Red
}

# Display next steps
Write-Host "\nNext steps:" -ForegroundColor Cyan
Write-Host "1. Review the generated plots in ../output/eda_plots/"
Write-Host "2. Check the console output for analysis results"
Write-Host "3. Commit your changes to the task-1 branch"
