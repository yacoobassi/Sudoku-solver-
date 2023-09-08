G20         ; Set units to inches (if not already)
G91         ; Set to incremental positioning
G1 X-0.04 F5 ; Move the X-axis by -0.04 inches (left) at a feedrate of 5 inches per minute
M5 S90
G4 P1
G90         ; Set to absolute positioning
G21         ; Set units to millimeters (if desired, replace with G20 for inches)
G92 X0 Y0   ; Set current position to X=0 and Y=0 (set as new home position)
