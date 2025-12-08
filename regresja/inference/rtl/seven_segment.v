/*
================================================================================
7-Segment Display Controller
================================================================================
Time-multiplexed display for two digits.
================================================================================
*/

module seven_segment_display (
    input wire clk,
    input wire rst,
    input wire [3:0] digit_left,   // Leftmost digit (from memory)
    input wire [3:0] digit_right,  // Rightmost digit (current)
    output reg [6:0] seg,           // Segments a-g (active low)
    output reg [3:0] an             // Anodes (active low)
);

    // Time-multiplexing counter for switching between digits
    // At 100 MHz, divide to ~1 kHz refresh rate (50,000 cycles per digit)
    localparam REFRESH_DIV = 17'd50000;
    
    reg [16:0] refresh_counter;
    reg digit_select;  // 0 = left, 1 = right
    
    // 7-segment encoding function
    function [6:0] seg_encode;
        input [3:0] digit;
        begin
            case (digit)
                4'd0: seg_encode = 7'b1000000;
                4'd1: seg_encode = 7'b1111001;
                4'd2: seg_encode = 7'b0100100;
                4'd3: seg_encode = 7'b0110000;
                4'd4: seg_encode = 7'b0011001;
                4'd5: seg_encode = 7'b0010010;
                4'd6: seg_encode = 7'b0000010;
                4'd7: seg_encode = 7'b1111000;
                4'd8: seg_encode = 7'b0000000;
                4'd9: seg_encode = 7'b0010000;
                default: seg_encode = 7'b0111111;  // Dash for invalid
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            refresh_counter <= 0;
            digit_select <= 0;
            an <= 4'b1111;  // All off
            seg <= 7'b1111111;
        end else begin
            // Increment refresh counter
            if (refresh_counter >= REFRESH_DIV - 1) begin
                refresh_counter <= 0;
                digit_select <= ~digit_select;  // Toggle between left and right
            end else begin
                refresh_counter <= refresh_counter + 1;
            end
            
            // Select which digit to display
            if (digit_select == 0) begin
                // Display leftmost digit (an[3])
                an <= 4'b0111;  // an[3] active (leftmost)
                seg <= seg_encode(digit_left);
            end else begin
                // Display rightmost digit (an[0])
                an <= 4'b1110;  // an[0] active (rightmost)
                seg <= seg_encode(digit_right);
            end
        end
    end

endmodule


