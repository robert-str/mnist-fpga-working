/*
================================================================================
UART Receiver Module (115200 baud compatible)
================================================================================
Universal UART RX module for receiving serial data.
================================================================================
*/

module uart_rx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input wire clk,
    input wire rst,
    input wire rx,
    output reg [7:0] data,
    output reg ready
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam STATE_IDLE=0, STATE_START=1, STATE_DATA=2, STATE_STOP=3;

    reg [1:0] state;
    reg [15:0] clk_cnt;
    reg [2:0] bit_cnt;
    reg [7:0] shift_reg;
    reg rx_sync1, rx_sync2;

    // Synchronize RX input (2-stage synchronizer for metastability)
    always @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;
    end

    // UART receive state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_IDLE;
            clk_cnt <= 0;
            bit_cnt <= 0;
            data <= 0;
            shift_reg <= 0;
            ready <= 0;
        end else begin
            ready <= 0;  // Default: ready is only high for 1 cycle
            
            case (state)
                // ----------------------------------------
                // IDLE: Wait for start bit (falling edge)
                // ----------------------------------------
                STATE_IDLE: begin
                    clk_cnt <= 0;
                    bit_cnt <= 0;
                    if (rx_sync2 == 0) begin
                        state <= STATE_START;
                    end
                end
                
                // ----------------------------------------
                // START: Verify start bit at middle
                // ----------------------------------------
                STATE_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2)) begin
                        if (rx_sync2 == 0) begin
                            // Valid start bit
                            clk_cnt <= 0;
                            state <= STATE_DATA;
                        end else begin
                            // False start (noise)
                            state <= STATE_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // DATA: Sample 8 data bits (LSB first)
                // ----------------------------------------
                STATE_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT) begin
                        clk_cnt <= 0;
                        shift_reg[bit_cnt] <= rx_sync2;
                        
                        if (bit_cnt == 7) begin
                            bit_cnt <= 0;
                            state <= STATE_STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                // ----------------------------------------
                // STOP: Wait for stop bit, output data
                // ----------------------------------------
                STATE_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT) begin
                        clk_cnt <= 0;
                        state <= STATE_IDLE;
                        data <= shift_reg;  // Transfer to output
                        ready <= 1;         // Signal data valid
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule


