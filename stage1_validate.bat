@echo off
REM Stage 1 Validation Script for Windows
REM This script runs the batch memory analysis and checks for errors

echo ===== Stage 1 Validation =====
echo Running validation tests for Stage 1 of the activation checkpointing project
echo Started at: %date% %time%
echo.

REM Create reports directory if it doesn't exist
if not exist reports mkdir reports

REM Step 1: Run batch memory analysis
echo Step 1: Running batch memory analysis...
conda run -n ml_env python starter_code/batch_memory_analysis.py --batch-sizes 4 8 16 32 64
if %ERRORLEVEL% neq 0 (
    echo ERROR: Batch memory analysis failed.
    exit /b 1
)
echo Batch memory analysis completed successfully.
echo.

REM Step 2: Run unit tests
echo Step 2: Running unit tests...
conda run -n ml_env python tests/test_profiler.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Unit tests failed.
    exit /b 1
)
echo Unit tests completed successfully.
echo.

REM Step 3: Verify that all required files are generated
echo Step 3: Verifying generated files...

set missing_files=0

REM Enable delayed expansion for variables in loops
setlocal enabledelayedexpansion

REM Check for CSV files
for %%b in (4 8 16 32 64) do (
    set node_csv=reports\profiler_stats_bs%%b_node_stats.csv
    set act_csv=reports\profiler_stats_bs%%b_activation_stats.csv
    
    if not exist !node_csv! (
        echo ERROR: Missing file: !node_csv!
        set /a missing_files+=1
    )
    
    if not exist !act_csv! (
        echo ERROR: Missing file: !act_csv!
        set /a missing_files+=1
    )
)

REM Check for plot files
set required_plots=reports\resnet152_batch_memory.png reports\resnet152_memory_vs_rank.png reports\resnet152_memory_breakdown.png reports\resnet152_latency_comparison.png

for %%p in (%required_plots%) do (
    if not exist %%p (
        echo ERROR: Missing file: %%p
        set /a missing_files+=1
    )
)

if %missing_files% equ 0 (
    echo All required files are present.
) else (
    echo ERROR: %missing_files% required files are missing.
    exit /b 1
)

REM Step 4: Summary
echo.
echo ===== Validation Summary =====
echo Batch memory analysis: PASSED
echo Unit tests: PASSED
echo File verification: PASSED
echo.
echo Stage 1 validation completed successfully!
echo Finished at: %date% %time%
echo.

exit /b 0