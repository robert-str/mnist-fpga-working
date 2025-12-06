/*
================================================================================
LeNet-5 Weight Loader via UART
================================================================================

Loads trained LeNet-5 CNN weights and biases from PC to FPGA BRAM.

Model: LeNet-5 Convolutional Neural Network for MNIST
  - Input:  28x28 image (1 channel)
  - Output: 10 classes (digits 0-9)

Layer Structure:
  - conv1: 6 filters, 5x5 kernel  -> weights: 150 bytes (6*1*5*5), biases: 24 bytes (6*4)
  - conv2: 16 filters, 5x5 kernel -> weights: 2400 bytes (16*6*5*5), biases: 64 bytes (16*4)
  - fc1:   120 neurons            -> weights: 48000 bytes (120*400), biases: 480 bytes (120*4)
  - fc2:   84 neurons             -> weights: 10080 bytes (84*120), biases: 336 bytes (84*4)
  - fc3:   10 neurons             -> weights: 840 bytes (10*84), biases: 40 bytes (10*4)

Total: 61470 bytes (weights) + 944 bytes (biases) + 4 bytes (shifts) = 62418 bytes

Protocol:
  - Start marker: 0xAA 0x55 (two bytes)
  - Data bytes: layer weights and biases in order
  - Shift values: 4 bytes (conv1_shift, conv2_shift, fc1_shift, fc2_shift)
  - End marker: 0x55 0xAA (two bytes)

Memory Layout:
  Address 0-149:        conv1 weights (150 bytes, 8-bit signed)
  Address 150-173:      conv1 biases (24 bytes = 6 x 4-byte)
  Address 174-2573:     conv2 weights (2400 bytes)
  Address 2574-2637:    conv2 biases (64 bytes = 16 x 4-byte)
  Address 2638-50637:   fc1 weights (48000 bytes)
  Address 50638-51117:  fc1 biases (480 bytes = 120 x 4-byte)
  Address 51118-61197:  fc2 weights (10080 bytes)
  Address 61198-61533:  fc2 biases (336 bytes = 84 x 4-byte)
  Address 61534-62373:  fc3 weights (840 bytes)
  Address 62374-62413:  fc3 biases (40 bytes = 10 x 4-byte)
  Address 62414-62417:  shift values (4 bytes: conv1, conv2, fc1, fc2)

LED Status:
  - led[0]:    Blinks when receiving bytes
  - led[1]:    HIGH when waiting for start marker
  - led[2]:    HIGH when receiving data
  - led[3]:    HIGH when transfer complete (success)
  - led[4]:    HIGH if error (overflow)
  - led[7:5]:  Unused
  - led[15:8]: Lower 8 bits of current address (progress indicator)

================================================================================
*/

// =============================================================================
// TOP MODULE - Use this for synthesis (only external pins)
// =============================================================================
module load_weights_top (
    input wire clk,              // 100 MHz System Clock
    input wire rst,              // Reset Button (active high)
    input wire rx,               // UART RX Line (from PC)
    output wire [15:0] led       // Debug LEDs
);

    // Internal signals for testing
    wire [15:0] weight_rd_addr;
    wire [7:0]  weight_rd_data;
    wire [7:0]  bias_rd_addr;
    wire [31:0] bias_rd_data;
    wire        transfer_done;
    
    // For testing: tie read addresses to 0
    assign weight_rd_addr = 16'd0;
    assign bias_rd_addr = 8'd0;

    // Instantiate the weight loader
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weight_rd_addr(weight_rd_addr),
        .weight_rd_data(weight_rd_data),
        .bias_rd_addr(bias_rd_addr),
        .bias_rd_data(bias_rd_data),
        .transfer_done(transfer_done),
        .led(led)
    );

endmodule


// =============================================================================
// WEIGHT LOADER MODULE - Has read ports for inference module
// =============================================================================
module weight_loader (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    
    // Read port for weights (inference module)
    input wire [15:0] weight_rd_addr,  // 0 to 61469 (16 bits needed)
    output reg [7:0]  weight_rd_data,  // 8-bit signed weight
    
    // Read port for biases (inference module)
    input wire [7:0]  bias_rd_addr,    // 0 to 235 (8 bits)
    output reg [31:0] bias_rd_data,    // 32-bit signed bias
    
    // Per-layer shift values for dynamic scaling
    output reg [4:0] shift_conv1,      // Shift for conv1 tanh
    output reg [4:0] shift_conv2,      // Shift for conv2 tanh
    output reg [4:0] shift_fc1,        // Shift for fc1 tanh
    output reg [4:0] shift_fc2,        // Shift for fc2 tanh
    
    // Status
    output reg transfer_done,
    
    // Debug LEDs
    output reg [15:0] led
);

    // Parameters
    parameter CLK_FREQ = 100_000_000;
    parameter BAUD_RATE = 115200;
    
    // Memory sizes for LeNet-5
    localparam CONV1_WEIGHTS = 150;      // 6*1*5*5
    localparam CONV1_BIASES  = 6;
    localparam CONV2_WEIGHTS = 2400;     // 16*6*5*5
    localparam CONV2_BIASES  = 16;
    localparam FC1_WEIGHTS   = 48000;    // 120*400
    localparam FC1_BIASES    = 120;
    localparam FC2_WEIGHTS   = 10080;    // 84*120
    localparam FC2_BIASES    = 84;
    localparam FC3_WEIGHTS   = 840;      // 10*84
    localparam FC3_BIASES    = 10;
    
    localparam TOTAL_WEIGHTS = CONV1_WEIGHTS + CONV2_WEIGHTS + FC1_WEIGHTS + FC2_WEIGHTS + FC3_WEIGHTS;  // 61470
    localparam TOTAL_BIASES  = CONV1_BIASES + CONV2_BIASES + FC1_BIASES + FC2_BIASES + FC3_BIASES;       // 236
    localparam TOTAL_BIAS_BYTES = TOTAL_BIASES * 4;  // 944
    localparam SHIFT_BYTES = 4;  // 4 shift values (conv1, conv2, fc1, fc2)
    localparam TOTAL_BYTES = TOTAL_WEIGHTS + TOTAL_BIAS_BYTES + SHIFT_BYTES;  // 62418
    
    // Address where shift values start
    localparam SHIFT_START_ADDR = TOTAL_WEIGHTS + TOTAL_BIAS_BYTES;  // 62414
    
    // Protocol markers
    localparam START_BYTE1 = 8'hAA;
    localparam START_BYTE2 = 8'h55;
    localparam END_BYTE1 = 8'h55;
    localparam END_BYTE2 = 8'hAA;
    
    // State machine states
    localparam STATE_WAIT_START1 = 3'd0;
    localparam STATE_WAIT_START2 = 3'd1;
    localparam STATE_RECEIVING   = 3'd2;
    localparam STATE_CHECK_END   = 3'd3;
    localparam STATE_DONE        = 3'd4;
    localparam STATE_ERROR       = 3'd5;

    // UART signals
    wire [7:0] rx_data;
    wire rx_ready;
    
    // Block RAM for weights (8-bit values) - needs 64KB
    (* ram_style = "block" *) reg [7:0] weight_bram [0:TOTAL_WEIGHTS-1];
    
    // Block RAM for biases (32-bit values)
    (* ram_style = "block" *) reg [31:0] bias_bram [0:TOTAL_BIASES-1];
    
    // State and control registers
    reg [2:0] state;
    reg [16:0] write_addr;       // Address for writing (0 to 62413)
    reg [7:0] prev_byte;         // Previous byte for end marker detection
    reg blink_toggle;            // For LED blinking
    
    // Bias assembly registers (4 bytes -> 32 bits)
    reg [1:0] bias_byte_cnt;     // Which byte of bias we're receiving (0-3)
    reg [31:0] bias_temp;        // Temporary register for assembling bias
    
    // UART Receiver Instance
    uart_rx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(rx_data),
        .ready(rx_ready)
    );

    // Synchronous read from weight BRAM
    always @(posedge clk) begin
        weight_rd_data <= weight_bram[weight_rd_addr];
    end
    
    // Synchronous read from bias BRAM
    always @(posedge clk) begin
        bias_rd_data <= bias_bram[bias_rd_addr];
    end
    
    // Calculate current layer boundaries for bias writing
    // This determines which bias index to write to based on write_addr
    function [7:0] get_bias_index;
        input [16:0] addr;
        reg [16:0] offset;
        begin
            // Check which layer we're in
            if (addr < CONV1_WEIGHTS) begin
                get_bias_index = 8'd0;  // In conv1 weights
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4) begin
                offset = addr - CONV1_WEIGHTS;
                get_bias_index = offset[9:2];  // bias index = offset / 4
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS) begin
                get_bias_index = 8'd0;  // In conv2 weights
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4) begin
                offset = addr - (CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS);
                get_bias_index = CONV1_BIASES + offset[9:2];
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS) begin
                get_bias_index = 8'd0;  // In fc1 weights
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4) begin
                offset = addr - (CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS);
                get_bias_index = CONV1_BIASES + CONV2_BIASES + offset[9:2];
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS) begin
                get_bias_index = 8'd0;  // In fc2 weights
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4) begin
                offset = addr - (CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS);
                get_bias_index = CONV1_BIASES + CONV2_BIASES + FC1_BIASES + offset[9:2];
            end else if (addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4 + FC3_WEIGHTS) begin
                get_bias_index = 8'd0;  // In fc3 weights
            end else begin
                offset = addr - (CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4 + FC3_WEIGHTS);
                get_bias_index = CONV1_BIASES + CONV2_BIASES + FC1_BIASES + FC2_BIASES + offset[9:2];
            end
        end
    endfunction
    
    // Check if current address is in weight region
    function is_weight_region;
        input [16:0] addr;
        begin
            is_weight_region = (addr < CONV1_WEIGHTS) ||
                              (addr >= CONV1_WEIGHTS + CONV1_BIASES*4 && 
                               addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS) ||
                              (addr >= CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 && 
                               addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS) ||
                              (addr >= CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 && 
                               addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS) ||
                              (addr >= CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4 && 
                               addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4 + FC3_WEIGHTS);
        end
    endfunction
    
    // Check if current address is in shift region (after all weights and biases)
    function is_shift_region;
        input [16:0] addr;
        begin
            is_shift_region = (addr >= SHIFT_START_ADDR && addr < SHIFT_START_ADDR + SHIFT_BYTES);
        end
    endfunction

    // Weight write address calculation - for PREVIOUS byte (write_addr - 1)
    // This is needed because we store prev_byte which corresponds to (write_addr - 1)
    reg [15:0] weight_wr_addr;
    wire [16:0] prev_addr = (write_addr > 0) ? write_addr - 1 : 0;
    
    always @(*) begin
        if (prev_addr < CONV1_WEIGHTS)
            weight_wr_addr = prev_addr[15:0];
        else if (prev_addr >= CONV1_WEIGHTS + CONV1_BIASES*4 && 
                 prev_addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS)
            weight_wr_addr = CONV1_WEIGHTS + (prev_addr - (CONV1_WEIGHTS + CONV1_BIASES*4));
        else if (prev_addr >= CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 && 
                 prev_addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS)
            weight_wr_addr = CONV1_WEIGHTS + CONV2_WEIGHTS + (prev_addr - (CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4));
        else if (prev_addr >= CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 && 
                 prev_addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS)
            weight_wr_addr = CONV1_WEIGHTS + CONV2_WEIGHTS + FC1_WEIGHTS + (prev_addr - (CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4));
        else if (prev_addr >= CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4 && 
                 prev_addr < CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4 + FC3_WEIGHTS)
            weight_wr_addr = CONV1_WEIGHTS + CONV2_WEIGHTS + FC1_WEIGHTS + FC2_WEIGHTS + (prev_addr - (CONV1_WEIGHTS + CONV1_BIASES*4 + CONV2_WEIGHTS + CONV2_BIASES*4 + FC1_WEIGHTS + FC1_BIASES*4 + FC2_WEIGHTS + FC2_BIASES*4));
        else
            weight_wr_addr = 16'd0;
    end

    // Main state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_WAIT_START1;
            write_addr <= 0;
            prev_byte <= 0;
            transfer_done <= 0;
            bias_byte_cnt <= 0;
            bias_temp <= 0;
            blink_toggle <= 0;
            led <= 16'b0000_0000_0000_0010;  // led[1] = waiting for start
            // Initialize shifts to match trained model values
            // From scale_info.txt: conv1=15, conv2=15, fc1=16, fc2=16
            shift_conv1 <= 5'd15;
            shift_conv2 <= 5'd15;
            shift_fc1 <= 5'd16;
            shift_fc2 <= 5'd16;
        end else begin
            
            case (state)
                // ============================================
                // Wait for first start byte (0xAA)
                // ============================================
                STATE_WAIT_START1: begin
                    led[1] <= 1;  // Waiting indicator
                    led[2] <= 0;
                    led[3] <= 0;
                    led[4] <= 0;
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        if (rx_data == START_BYTE1) begin
                            state <= STATE_WAIT_START2;
                        end
                    end
                end
                
                // ============================================
                // Wait for second start byte (0x55)
                // ============================================
                STATE_WAIT_START2: begin
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        if (rx_data == START_BYTE2) begin
                            // Valid start sequence received
                            state <= STATE_RECEIVING;
                            write_addr <= 0;
                            bias_byte_cnt <= 0;
                            bias_temp <= 0;
                            led[1] <= 0;
                            led[2] <= 1;  // Receiving indicator
                        end else if (rx_data == START_BYTE1) begin
                            // Another 0xAA, stay in this state
                            state <= STATE_WAIT_START2;
                        end else begin
                            // Invalid sequence, go back to waiting
                            state <= STATE_WAIT_START1;
                        end
                    end
                end
                
                // ============================================
                // Receiving data bytes
                // ============================================
                STATE_RECEIVING: begin
                    led[2] <= 1;
                    led[15:8] <= write_addr[7:0];  // Show progress
                    
                    if (rx_ready) begin
                        blink_toggle <= ~blink_toggle;
                        led[0] <= blink_toggle;
                        
                        // Store the previous byte if we have data pending
                        if (write_addr > 0 && write_addr <= TOTAL_BYTES) begin
                            if (is_shift_region(write_addr - 1)) begin
                                // Storing shift values (4 bytes: conv1, conv2, fc1, fc2)
                                case (write_addr - 1 - SHIFT_START_ADDR)
                                    0: shift_conv1 <= prev_byte[4:0];
                                    1: shift_conv2 <= prev_byte[4:0];
                                    2: shift_fc1   <= prev_byte[4:0];
                                    3: shift_fc2   <= prev_byte[4:0];
                                endcase
                            end else if (is_weight_region(write_addr - 1)) begin
                                // Storing weight (weight_wr_addr already computed for prev_addr)
                                weight_bram[weight_wr_addr] <= prev_byte;
                            end else begin
                                // Storing bias byte (little-endian: first byte = LSB)
                                // Accumulate bytes 0,1,2 in bias_temp, then write on byte 3
                                case (bias_byte_cnt)
                                    2'd0: bias_temp[7:0]   <= prev_byte;  // LSB (first byte)
                                    2'd1: bias_temp[15:8]  <= prev_byte;
                                    2'd2: bias_temp[23:16] <= prev_byte;
                                    2'd3: begin
                                        // Write complete bias to BRAM (prev_byte is MSB)
                                        bias_bram[get_bias_index(write_addr - 1)] <= 
                                            {prev_byte, bias_temp[23:16], bias_temp[15:8], bias_temp[7:0]};
                                    end
                                endcase
                                bias_byte_cnt <= bias_byte_cnt + 1;
                            end
                        end
                        
                        // Check for end marker after storing
                        if (prev_byte == END_BYTE1 && rx_data == END_BYTE2) begin
                            // End marker detected!
                            state <= STATE_DONE;
                            transfer_done <= 1;
                            led[2] <= 0;
                            led[3] <= 1;  // Success indicator
                        end else begin
                            // Check for overflow
                            if (write_addr >= TOTAL_BYTES + 1) begin
                                state <= STATE_ERROR;
                                led[4] <= 1;
                            end else begin
                                write_addr <= write_addr + 1;
                                prev_byte <= rx_data;
                            end
                        end
                    end
                end
                
                // ============================================
                // Transfer complete
                // ============================================
                STATE_DONE: begin
                    led[3] <= 1;  // Success - stays on
                    led[15:8] <= 8'hFF;  // All progress LEDs on
                    transfer_done <= 1;
                    // Stay in this state until reset
                end
                
                // ============================================
                // Error state
                // ============================================
                STATE_ERROR: begin
                    led[4] <= 1;  // Error indicator stays on
                    // Stay in this state until reset
                end
                
                default: begin
                    state <= STATE_WAIT_START1;
                end
            endcase
        end
    end

endmodule


// =============================================================================
// UART Receiver Module (115200 baud compatible)
// =============================================================================
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
    
    // State definitions
    localparam STATE_IDLE  = 2'd0;
    localparam STATE_START = 2'd1;
    localparam STATE_DATA  = 2'd2;
    localparam STATE_STOP  = 2'd3;

    reg [1:0] state;
    reg [15:0] clk_cnt;           // Clock counter (16 bits for flexibility)
    reg [2:0] bit_cnt;            // Bit counter (0-7)
    reg [7:0] rx_shift;           // Shift register for incoming bits
    reg rx_sync1, rx_sync2;       // Double-flop synchronizer

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
            rx_shift <= 0;
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
                        rx_shift[bit_cnt] <= rx_sync2;
                        
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
                        data <= rx_shift;  // Transfer to output
                        ready <= 1;        // Signal data valid
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
