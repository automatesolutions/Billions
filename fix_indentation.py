#!/usr/bin/env python3
# Fix indentation issue in SPS.py

with open('funda/SPS.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 656 (index 655) - ensure proper indentation
if len(lines) > 655:
    # Check if the line needs fixing
    line = lines[655]
    if line.strip().startswith('html.Div([') and not line.startswith('    '):
        lines[655] = '    html.Div([\n'
        print("Fixed indentation on line 656")
    else:
        print("Line 656 already has correct indentation")

with open('funda/SPS.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("File updated successfully")
