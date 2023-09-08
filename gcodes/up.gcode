G20         ; Set units to inches (if not already)
G91         ; Set to incremental positioning
G1 Y0.04 F5 ; Move the Y-axis by 0.06 inches at a feedrate of 5 inches per minute
G90         ; Set to absolute positioning
G21         ; Set units to millimeters (if desired, replace with G20 for inches)

 ; Set feedrate to 2000 (adjust as needed)

G92 X0 Y0   ; Set current position to X=0 and Y=0 (set as new home position)
