/*
================================================================================
LeNet-5 CNN Inference Module - Complete Implementation
================================================================================

Architecture:
  Input: 28x28 grayscale image (8-bit signed, normalized)
  
  Conv1:  6 filters, 5x5, padding=2 -> 28x28x6, Tanh
  Pool1:  Average 2x2, stride 2     -> 14x14x6
  Conv2:  16 filters, 5x5           -> 10x10x16, Tanh  
  Pool2:  Average 2x2, stride 2     -> 5x5x16 = 400
  FC1:    400 -> 120, Tanh
  FC2:    120 -> 84, Tanh
  FC3:    84 -> 10 (logits)
  Output: argmax -> digit 0-9

Weight Memory (8-bit):
  [0:149]       conv1 (6*1*5*5=150)
  [150:2549]    conv2 (16*6*5*5=2400)
  [2550:50549]  fc1 (120*400=48000)
  [50550:60629] fc2 (84*120=10080)
  [60630:61469] fc3 (10*84=840)

Bias Memory (32-bit):
  [0:5]     conv1 (6)
  [6:21]    conv2 (16)
  [22:141]  fc1 (120)
  [142:225] fc2 (84)
  [226:235] fc3 (10)

================================================================================
*/

module inference (
    input wire clk,
    input wire rst,
    
    // Weight memory interface
    output reg [15:0] weight_addr,
    input wire [7:0]  weight_data,
    
    // Bias memory interface  
    output reg [7:0]  bias_addr,
    input wire [31:0] bias_data,
    
    // Per-layer shift values for dynamic scaling
    input wire [4:0]  shift_conv1,
    input wire [4:0]  shift_conv2,
    input wire [4:0]  shift_fc1,
    input wire [4:0]  shift_fc2,
    
    // Control
    input wire        weights_ready,
    input wire        start_inference,
    input wire [7:0]  input_pixel,
    output reg [9:0]  input_addr,
    
    // Outputs
    output reg [3:0]  predicted_digit,
    output reg        inference_done,
    output reg        busy
);

    // =========================================================================
    // Parameters
    // =========================================================================
    
    // Layer sizes
    localparam INPUT_W = 28, INPUT_H = 28;
    localparam CONV1_F = 6, CONV1_K = 5;
    localparam POOL1_W = 14, POOL1_H = 14;
    localparam CONV2_F = 16, CONV2_K = 5, CONV2_C = 6;
    localparam POOL2_W = 5, POOL2_H = 5;
    localparam FC1_IN = 400, FC1_OUT = 120;
    localparam FC2_IN = 120, FC2_OUT = 84;
    localparam FC3_IN = 84, FC3_OUT = 10;
    
    // Weight offsets
    localparam W_CONV1 = 0;
    localparam W_CONV2 = 150;
    localparam W_FC1 = 2550;
    localparam W_FC2 = 50550;
    localparam W_FC3 = 60630;
    
    // Bias offsets
    localparam B_CONV1 = 0;
    localparam B_CONV2 = 6;
    localparam B_FC1 = 22;
    localparam B_FC2 = 142;
    localparam B_FC3 = 226;

    // =========================================================================
    // State machine
    // =========================================================================
    localparam S_IDLE = 0, S_LOAD_IMG = 1;
    localparam S_CONV1 = 2, S_POOL1 = 3;
    localparam S_CONV2 = 4, S_POOL2 = 5;
    localparam S_FC1 = 6, S_FC2 = 7, S_FC3 = 8;
    localparam S_ARGMAX = 9, S_DONE = 10;
    
    reg [3:0] state;
    reg [3:0] step;  // Sub-step within state

    // =========================================================================
    // Memory buffers
    // =========================================================================
    
    // Input image (8-bit signed)
    (* ram_style = "block" *) reg signed [7:0] img [0:783];
    
    // Pool1 output: 14*14*6 = 1176 (16-bit)
    (* ram_style = "block" *) reg signed [15:0] pool1 [0:1175];
    
    // Pool2 output: 5*5*16 = 400 (16-bit)
    (* ram_style = "block" *) reg signed [15:0] pool2 [0:399];
    
    // FC outputs
    (* ram_style = "block" *) reg signed [15:0] fc1_out [0:119];
    (* ram_style = "block" *) reg signed [15:0] fc2_out [0:83];
    
    // Conv row buffers for pooling (store 2 rows)
    // Conv1: 28*6*2 = 336, Conv2: 10*16*2 = 320
    (* ram_style = "block" *) reg signed [15:0] conv_buf [0:335];
    
    // Final scores
    reg signed [31:0] scores [0:9];

    // =========================================================================
    // Processing registers
    // =========================================================================
    reg [9:0] cnt;                    // General counter
    reg [4:0] out_x, out_y;           // Output position
    reg [4:0] filt;                   // Filter index
    reg signed [3:0] kx, ky;          // Kernel position (needs 4 bits for 0-4 range)
    reg [3:0] ch;                     // Channel
    reg [8:0] neuron;                 // FC neuron
    reg [9:0] in_idx;                 // FC input index
    
    reg signed [31:0] acc;            // Accumulator
    reg signed [31:0] max_val;
    reg [3:0] max_idx;
    
    // Pipeline registers
    reg signed [7:0] w_pipe;          // Weight pipeline
    reg signed [15:0] in_pipe;        // Input pipeline
    reg signed [23:0] prod;           // Product
    reg [1:0] pipe;                   // Pipeline stage
    
    // Computed addresses
    reg signed [5:0] ix, iy;          // Image x,y with padding
    reg [9:0] buf_idx;                // Buffer index
    
    // Row tracking for pooling
    reg row_parity;                   // 0 or 1 for which row buffer half

    // =========================================================================
    // Tanh approximation with per-layer dynamic shift
    // Shift amount varies per layer based on quantization scale
    // shifted_acc is used as an intermediate blocking-assignment variable
    // in the sequential always block below
    // =========================================================================
    reg signed [31:0] shifted_acc;

    // =========================================================================
    // Main state machine
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            step <= 0;
            busy <= 0;
            inference_done <= 0;
            predicted_digit <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
            input_addr <= 0;
            cnt <= 0;
            pipe <= 0;
            acc <= 0;
            max_val <= 32'sh80000000;
            max_idx <= 0;
            out_x <= 0; out_y <= 0;
            filt <= 0;
            kx <= 0; ky <= 0;
            ch <= 0;
            neuron <= 0;
            in_idx <= 0;
            row_parity <= 0;
        end else begin
            inference_done <= 0;
            
            case (state)
            
            // =================================================================
            // IDLE
            // =================================================================
            S_IDLE: begin
                busy <= 0;
                if (start_inference && weights_ready) begin
                    state <= S_LOAD_IMG;
                    step <= 0;
                    cnt <= 0;
                    busy <= 1;
                end
            end
            
            // =================================================================
            // LOAD IMAGE: Read 784 pixels
            // =================================================================
            S_LOAD_IMG: begin
                case (step)
                    0: begin  // Set address
                        input_addr <= cnt;
                        step <= 1;
                    end
                    1: begin  // Wait for data
                        step <= 2;
                    end
                    2: begin  // Store pixel
                        img[cnt] <= $signed(input_pixel);
                        if (cnt < 783) begin
                            cnt <= cnt + 1;
                            step <= 0;
                        end else begin
                            // Start Conv1
                            state <= S_CONV1;
                            step <= 0;
                            out_x <= 0;
                            out_y <= 0;
                            filt <= 0;
                            row_parity <= 0;
                        end
                    end
                endcase
            end
            
            // =================================================================
            // CONV1: 6 filters, 5x5, padding=2 -> 28x28x6
            // Process one pixel at a time, store in conv_buf for pooling
            // =================================================================
            S_CONV1: begin
                case (step)
                    0: begin  // Load bias
                        bias_addr <= B_CONV1 + filt;
                        step <= 1;
                    end
                    1: begin  // Wait for bias
                        step <= 2;
                    end
                    2: begin  // Initialize accumulator with bias
                        acc <= $signed(bias_data);
                        kx <= -2;
                        ky <= -2;
                        weight_addr <= W_CONV1 + filt * 25;
                        step <= 3;
                    end
                    3: begin  // Kernel loop - set up addresses
                        ix <= $signed({1'b0, out_x}) + kx;
                        iy <= $signed({1'b0, out_y}) + ky;
                        step <= 4;
                    end
                    4: begin  // Wait for weight, get input
                        w_pipe <= $signed(weight_data);
                        // Padded read
                        if (ix < 0 || ix >= INPUT_W || iy < 0 || iy >= INPUT_H)
                            in_pipe <= 0;
                        else
                            in_pipe <= {{8{img[iy*INPUT_W + ix][7]}}, img[iy*INPUT_W + ix]};
                        step <= 5;
                    end
                    5: begin  // MAC
                        prod <= w_pipe * $signed(in_pipe[7:0]);
                        step <= 6;
                    end
                    6: begin  // Accumulate
                        acc <= acc + {{8{prod[23]}}, prod};
                        
                        // Next kernel position
                        if (kx < 2) begin
                            kx <= kx + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else if (ky < 2) begin
                            kx <= -2;
                            ky <= ky + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else begin
                            step <= 7;  // Done with kernel
                        end
                    end
                    7: begin  // Apply tanh with layer-specific shift, store in conv_buf
                        // Index: row_parity * 168 + out_x * 6 + filt
                        // (168 = 28 * 6, one row of all filters)
                        // Use shift_conv1 for dynamic scaling
                        shifted_acc = acc >>> shift_conv1;
                        if (shifted_acc > 127)
                            conv_buf[row_parity * 168 + out_x * CONV1_F + filt] <= 127;
                        else if (shifted_acc < -127)
                            conv_buf[row_parity * 168 + out_x * CONV1_F + filt] <= -127;
                        else
                            conv_buf[row_parity * 168 + out_x * CONV1_F + filt] <= shifted_acc[15:0];
                        step <= 8;
                    end
                    8: begin  // Next position
                        if (filt < CONV1_F - 1) begin
                            filt <= filt + 1;
                            step <= 0;
                        end else begin
                            filt <= 0;
                            if (out_x < INPUT_W - 1) begin
                                out_x <= out_x + 1;
                                step <= 0;
                            end else begin
                                out_x <= 0;
                                // Row complete - check if we have 2 rows for pooling
                                if (out_y[0] == 1) begin
                                    // Odd row (1, 3, 5...) - two rows ready
                                    state <= S_POOL1;
                                    step <= 0;
                                    row_parity <= 0;
                                end else begin
                                    // Even row (0, 2, 4...) - need one more row
                                    row_parity <= 1;
                                    out_y <= out_y + 1;
                                    step <= 0;
                                end
                            end
                        end
                    end
                endcase
            end
            
            // =================================================================
            // POOL1: Average 2x2 -> 14x14x6
            // =================================================================
            S_POOL1: begin
                case (step)
                    0: begin  // Initialize pooling
                        out_x <= 0;
                        filt <= 0;
                        step <= 1;
                    end
                    1: begin  // Compute average of 2x2 block
                        // Indices in conv_buf:
                        // Top-left:     0 * 168 + (out_x*2) * 6 + filt
                        // Top-right:    0 * 168 + (out_x*2+1) * 6 + filt
                        // Bottom-left:  1 * 168 + (out_x*2) * 6 + filt
                        // Bottom-right: 1 * 168 + (out_x*2+1) * 6 + filt
                        buf_idx <= out_x * 2 * CONV1_F + filt;
                        step <= 2;
                    end
                    2: begin  // Sum and store
                        // Average = (tl + tr + bl + br) / 4
                        acc <= (conv_buf[buf_idx] + 
                               conv_buf[buf_idx + CONV1_F] + 
                               conv_buf[168 + buf_idx] + 
                               conv_buf[168 + buf_idx + CONV1_F]) >>> 2;
                        step <= 3;
                    end
                    3: begin  // Store result
                        // Pool1 index: (out_y/2) * 14 * 6 + out_x * 6 + filt
                        pool1[(out_y >> 1) * POOL1_W * CONV1_F + out_x * CONV1_F + filt] <= acc[15:0];
                        
                        if (filt < CONV1_F - 1) begin
                            filt <= filt + 1;
                            step <= 1;
                        end else begin
                            filt <= 0;
                            if (out_x < POOL1_W - 1) begin
                                out_x <= out_x + 1;
                                step <= 1;
                            end else begin
                                // Pooling row done
                                if (out_y >= INPUT_H - 1) begin
                                    // Conv1 + Pool1 complete (processed all 28 rows)
                                    state <= S_CONV2;
                                    step <= 0;
                                    out_x <= 0;
                                    out_y <= 0;
                                    filt <= 0;
                                    row_parity <= 0;
                                end else begin
                                    // Continue conv1 with next row
                                    state <= S_CONV1;
                                    step <= 0;
                                    out_y <= out_y + 1;
                                    row_parity <= 0;
                                end
                            end
                        end
                    end
                endcase
            end
            
            // =================================================================
            // CONV2: 16 filters, 6 channels, 5x5 -> 10x10x16
            // Input: pool1 (14x14x6), Output: 10x10x16
            // =================================================================
            S_CONV2: begin
                case (step)
                    0: begin  // Load bias
                        bias_addr <= B_CONV2 + filt;
                        step <= 1;
                    end
                    1: begin
                        step <= 2;
                    end
                    2: begin  // Init accumulator
                        acc <= $signed(bias_data);
                        ch <= 0;
                        kx <= 0;
                        ky <= 0;
                        weight_addr <= W_CONV2 + filt * CONV2_C * 25;
                        step <= 3;
                    end
                    3: begin  // Kernel loop - compute pool1 index
                        ix <= out_x + kx;
                        iy <= out_y + ky;
                        step <= 4;
                    end
                    4: begin  // Get weight and input
                        w_pipe <= $signed(weight_data);
                        // Pool1 index: iy * 14 * 6 + ix * 6 + ch
                        in_pipe <= pool1[iy * POOL1_W * CONV2_C + ix * CONV2_C + ch];
                        step <= 5;
                    end
                    5: begin  // MAC
                        prod <= w_pipe * $signed(in_pipe[7:0]);
                        step <= 6;
                    end
                    6: begin  // Accumulate
                        acc <= acc + {{8{prod[23]}}, prod};
                        
                        if (kx < CONV2_K - 1) begin
                            kx <= kx + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else if (ky < CONV2_K - 1) begin
                            kx <= 0;
                            ky <= ky + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else if (ch < CONV2_C - 1) begin
                            kx <= 0;
                            ky <= 0;
                            ch <= ch + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else begin
                            step <= 7;
                        end
                    end
                    7: begin  // Apply tanh with layer-specific shift, store
                        // Conv_buf: row_parity * 160 + out_x * 16 + filt
                        // Use shift_conv2 for dynamic scaling
                        shifted_acc = acc >>> shift_conv2;
                        if (shifted_acc > 127)
                            conv_buf[row_parity * 160 + out_x * CONV2_F + filt] <= 127;
                        else if (shifted_acc < -127)
                            conv_buf[row_parity * 160 + out_x * CONV2_F + filt] <= -127;
                        else
                            conv_buf[row_parity * 160 + out_x * CONV2_F + filt] <= shifted_acc[15:0];
                        step <= 8;
                    end
                    8: begin  // Next position
                        if (filt < CONV2_F - 1) begin
                            filt <= filt + 1;
                            step <= 0;
                        end else begin
                            filt <= 0;
                            if (out_x < 10 - 1) begin
                                out_x <= out_x + 1;
                                step <= 0;
                            end else begin
                                out_x <= 0;
                                // Check if we have 2 rows for pooling
                                if (out_y[0] == 1) begin
                                    // Odd row - two rows ready
                                    state <= S_POOL2;
                                    step <= 0;
                                    row_parity <= 0;
                                end else begin
                                    // Even row - need one more
                                    row_parity <= 1;
                                    out_y <= out_y + 1;
                                    step <= 0;
                                end
                            end
                        end
                    end
                endcase
            end
            
            // =================================================================
            // POOL2: Average 2x2 -> 5x5x16 = 400
            // =================================================================
            S_POOL2: begin
                case (step)
                    0: begin
                        out_x <= 0;
                        filt <= 0;
                        step <= 1;
                    end
                    1: begin
                        buf_idx <= out_x * 2 * CONV2_F + filt;
                        step <= 2;
                    end
                    2: begin
                        acc <= (conv_buf[buf_idx] + 
                               conv_buf[buf_idx + CONV2_F] + 
                               conv_buf[160 + buf_idx] + 
                               conv_buf[160 + buf_idx + CONV2_F]) >>> 2;
                        step <= 3;
                    end
                    3: begin
                        // Pool2 index: filt * 25 + row * 5 + col (NCHW order to match PyTorch flatten)
                        pool2[filt * POOL2_W * POOL2_H + (out_y >> 1) * POOL2_W + out_x] <= acc[15:0];
                        
                        if (filt < CONV2_F - 1) begin
                            filt <= filt + 1;
                            step <= 1;
                        end else begin
                            filt <= 0;
                            if (out_x < POOL2_W - 1) begin
                                out_x <= out_x + 1;
                                step <= 1;
                            end else begin
                                // Check if conv2 is complete (all 10 rows)
                                if (out_y >= 10 - 1) begin
                                    state <= S_FC1;
                                    step <= 0;
                                    neuron <= 0;
                                end else begin
                                    // Continue conv2
                                    state <= S_CONV2;
                                    step <= 0;
                                    out_y <= out_y + 1;
                                    row_parity <= 0;
                                end
                            end
                        end
                    end
                endcase
            end
            
            // =================================================================
            // FC1: 400 -> 120
            // =================================================================
            S_FC1: begin
                case (step)
                    0: begin  // Load bias
                        bias_addr <= B_FC1 + neuron;
                        step <= 1;
                    end
                    1: begin
                        step <= 2;
                    end
                    2: begin
                        acc <= $signed(bias_data);
                        in_idx <= 0;
                        weight_addr <= W_FC1 + neuron * FC1_IN;
                        step <= 3;
                    end
                    3: begin  // Set up weight address
                        step <= 4;
                    end
                    4: begin  // Get weight and input
                        w_pipe <= $signed(weight_data);
                        in_pipe <= pool2[in_idx];
                        step <= 5;
                    end
                    5: begin  // MAC
                        prod <= w_pipe * $signed(in_pipe[7:0]);
                        step <= 6;
                    end
                    6: begin  // Accumulate
                        acc <= acc + {{8{prod[23]}}, prod};
                        
                        if (in_idx < FC1_IN - 1) begin
                            in_idx <= in_idx + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else begin
                            step <= 7;
                        end
                    end
                    7: begin  // Store with tanh using layer-specific shift
                        // Use shift_fc1 for dynamic scaling
                        shifted_acc = acc >>> shift_fc1;
                        if (shifted_acc > 127)
                            fc1_out[neuron] <= 127;
                        else if (shifted_acc < -127)
                            fc1_out[neuron] <= -127;
                        else
                            fc1_out[neuron] <= shifted_acc[15:0];
                        
                        if (neuron < FC1_OUT - 1) begin
                            neuron <= neuron + 1;
                            step <= 0;
                        end else begin
                            state <= S_FC2;
                            step <= 0;
                            neuron <= 0;
                        end
                    end
                endcase
            end
            
            // =================================================================
            // FC2: 120 -> 84
            // =================================================================
            S_FC2: begin
                case (step)
                    0: begin
                        bias_addr <= B_FC2 + neuron;
                        step <= 1;
                    end
                    1: begin
                        step <= 2;
                    end
                    2: begin
                        acc <= $signed(bias_data);
                        in_idx <= 0;
                        weight_addr <= W_FC2 + neuron * FC2_IN;
                        step <= 3;
                    end
                    3: begin
                        step <= 4;
                    end
                    4: begin
                        w_pipe <= $signed(weight_data);
                        in_pipe <= fc1_out[in_idx];
                        step <= 5;
                    end
                    5: begin
                        prod <= w_pipe * $signed(in_pipe[7:0]);
                        step <= 6;
                    end
                    6: begin
                        acc <= acc + {{8{prod[23]}}, prod};
                        
                        if (in_idx < FC2_IN - 1) begin
                            in_idx <= in_idx + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else begin
                            step <= 7;
                        end
                    end
                    7: begin  // Store with tanh using layer-specific shift
                        // Use shift_fc2 for dynamic scaling
                        shifted_acc = acc >>> shift_fc2;
                        if (shifted_acc > 127)
                            fc2_out[neuron] <= 127;
                        else if (shifted_acc < -127)
                            fc2_out[neuron] <= -127;
                        else
                            fc2_out[neuron] <= shifted_acc[15:0];
                        
                        if (neuron < FC2_OUT - 1) begin
                            neuron <= neuron + 1;
                            step <= 0;
                        end else begin
                            state <= S_FC3;
                            step <= 0;
                            neuron <= 0;
                        end
                    end
                endcase
            end
            
            // =================================================================
            // FC3: 84 -> 10 (no activation)
            // =================================================================
            S_FC3: begin
                case (step)
                    0: begin
                        bias_addr <= B_FC3 + neuron;
                        step <= 1;
                    end
                    1: begin
                        step <= 2;
                    end
                    2: begin
                        acc <= $signed(bias_data);
                        in_idx <= 0;
                        weight_addr <= W_FC3 + neuron * FC3_IN;
                        step <= 3;
                    end
                    3: begin
                        step <= 4;
                    end
                    4: begin
                        w_pipe <= $signed(weight_data);
                        in_pipe <= fc2_out[in_idx];
                        step <= 5;
                    end
                    5: begin
                        prod <= w_pipe * $signed(in_pipe[7:0]);
                        step <= 6;
                    end
                    6: begin
                        acc <= acc + {{8{prod[23]}}, prod};
                        
                        if (in_idx < FC3_IN - 1) begin
                            in_idx <= in_idx + 1;
                            weight_addr <= weight_addr + 1;
                            step <= 3;
                        end else begin
                            step <= 7;
                        end
                    end
                    7: begin  // Store raw score
                        scores[neuron] <= acc;
                        
                        if (neuron < FC3_OUT - 1) begin
                            neuron <= neuron + 1;
                            step <= 0;
                        end else begin
                            state <= S_ARGMAX;
                            step <= 0;
                            neuron <= 0;
                            max_val <= 32'sh80000000;
                            max_idx <= 0;
                        end
                    end
                endcase
            end
            
            // =================================================================
            // ARGMAX
            // =================================================================
            S_ARGMAX: begin
                if (neuron < FC3_OUT) begin
                    if (scores[neuron] > max_val) begin
                        max_val <= scores[neuron];
                        max_idx <= neuron[3:0];
                    end
                    neuron <= neuron + 1;
                end else begin
                    predicted_digit <= max_idx;
                    state <= S_DONE;
                end
            end
            
            // =================================================================
            // DONE
            // =================================================================
            S_DONE: begin
                inference_done <= 1;
                busy <= 0;
                state <= S_IDLE;
            end
            
            default: state <= S_IDLE;
            
            endcase
        end
    end

endmodule


// =============================================================================
// TOP MODULE
// =============================================================================
module lenet5_top (
    input wire clk,
    input wire rst,
    input wire rx,
    output wire tx,
    output wire [15:0] led,
    output wire [6:0] seg,
    output wire [3:0] an,
    input wire [15:0] sw,
    input wire btnU, btnD, btnL, btnR
);

    wire [15:0] weight_rd_addr;
    wire [7:0]  weight_rd_data;
    wire [7:0]  bias_rd_addr;
    wire [31:0] bias_rd_data;
    wire        weights_loaded;
    wire [15:0] loader_led;
    
    // Per-layer shift values for dynamic scaling
    wire [4:0]  shift_conv1;
    wire [4:0]  shift_conv2;
    wire [4:0]  shift_fc1;
    wire [4:0]  shift_fc2;
    
    wire [15:0] inf_weight_addr;
    wire [7:0]  inf_bias_addr;
    wire [9:0]  inf_input_addr;
    wire [7:0]  inf_input_pixel;
    wire [3:0]  predicted_digit;
    wire        inference_done;
    wire        inference_busy;
    
    wire [9:0]  img_wr_addr;
    wire [7:0]  img_wr_data;
    wire        img_wr_en;
    wire        img_loaded;
    
    wire start_inference_pulse;
    reg [15:0] led_reg;
    reg [3:0] display_digit;
    
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weight_rd_addr(weight_rd_addr),
        .weight_rd_data(weight_rd_data),
        .bias_rd_addr(bias_rd_addr),
        .bias_rd_data(bias_rd_data),
        .shift_conv1(shift_conv1),
        .shift_conv2(shift_conv2),
        .shift_fc1(shift_fc1),
        .shift_fc2(shift_fc2),
        .transfer_done(weights_loaded),
        .led(loader_led)
    );
    
    image_loader u_image_loader (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weights_loaded(weights_loaded),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .image_loaded(img_loaded)
    );
    
    image_ram u_image_ram (
        .clk(clk),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .rd_addr(inf_input_addr),
        .rd_data(inf_input_pixel)
    );
    
    assign weight_rd_addr = inf_weight_addr;
    assign bias_rd_addr = inf_bias_addr;
    
    inference u_inference (
        .clk(clk),
        .rst(rst),
        .weight_addr(inf_weight_addr),
        .weight_data(weight_rd_data),
        .bias_addr(inf_bias_addr),
        .bias_data(bias_rd_data),
        .shift_conv1(shift_conv1),
        .shift_conv2(shift_conv2),
        .shift_fc1(shift_fc1),
        .shift_fc2(shift_fc2),
        .weights_ready(weights_loaded),
        .start_inference(start_inference_pulse),
        .input_pixel(inf_input_pixel),
        .input_addr(inf_input_addr),
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(inference_busy)
    );
    
    reg img_loaded_prev;
    always @(posedge clk) begin
        if (rst)
            img_loaded_prev <= 0;
        else
            img_loaded_prev <= img_loaded;
    end
    
    assign start_inference_pulse = img_loaded && !img_loaded_prev;
    
    always @(posedge clk) begin
        if (rst) begin
            led_reg <= 0;
            display_digit <= 0;
        end else begin
            if (!weights_loaded) begin
                led_reg <= loader_led;
            end else begin
                led_reg[3:0] <= predicted_digit;
                led_reg[4] <= inference_busy;
                led_reg[5] <= inference_done;
                led_reg[6] <= img_loaded;
                led_reg[7] <= weights_loaded;
                led_reg[15:8] <= loader_led[15:8];
            end
            
            if (inference_done)
                display_digit <= predicted_digit;
        end
    end
    
    assign led = led_reg;
    
    seven_segment_display u_display (
        .clk(clk),
        .rst(rst),
        .digit(display_digit),
        .seg(seg),
        .an(an)
    );
    
    assign tx = 1'b1;

endmodule


// =============================================================================
// Image Loader
// =============================================================================
module image_loader (
    input wire clk,
    input wire rst,
    input wire rx,
    input wire weights_loaded,
    output reg [9:0] wr_addr,
    output reg [7:0] wr_data,
    output reg wr_en,
    output reg image_loaded
);

    localparam IMG_START1 = 8'hBB;
    localparam IMG_START2 = 8'h66;
    localparam IMG_END1 = 8'h66;
    localparam IMG_END2 = 8'hBB;
    localparam IMG_SIZE = 784;

    localparam ST_WAIT1 = 0, ST_WAIT2 = 1, ST_RECV = 2, ST_END = 3, ST_DONE = 4;

    wire [7:0] rx_data;
    wire rx_ready;
    
    uart_rx #(.CLK_FREQ(100_000_000), .BAUD_RATE(115200)) u_rx (
        .clk(clk), .rst(rst), .rx(rx), .data(rx_data), .ready(rx_ready)
    );
    
    reg [2:0] state;
    reg [9:0] byte_cnt;

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_WAIT1;
            wr_addr <= 0;
            wr_data <= 0;
            wr_en <= 0;
            byte_cnt <= 0;
            image_loaded <= 0;
        end else begin
            wr_en <= 0;
            image_loaded <= 0;
            
            if (weights_loaded) begin
                case (state)
                    ST_WAIT1: if (rx_ready && rx_data == IMG_START1) state <= ST_WAIT2;
                    
                    ST_WAIT2: if (rx_ready) begin
                        if (rx_data == IMG_START2) begin
                            state <= ST_RECV;
                            byte_cnt <= 0;
                        end else if (rx_data != IMG_START1)
                            state <= ST_WAIT1;
                    end
                    
                    ST_RECV: if (rx_ready) begin
                        wr_addr <= byte_cnt;
                        wr_data <= rx_data;
                        wr_en <= 1;
                        if (byte_cnt >= IMG_SIZE - 1) begin
                            state <= ST_END;
                            byte_cnt <= 0;
                        end else
                            byte_cnt <= byte_cnt + 1;
                    end
                    
                    ST_END: if (rx_ready) begin
                        if (byte_cnt == 0 && rx_data == IMG_END1)
                            byte_cnt <= 1;
                        else if (byte_cnt == 1 && rx_data == IMG_END2) begin
                            state <= ST_DONE;
                            image_loaded <= 1;
                        end else
                            byte_cnt <= 0;
                    end
                    
                    ST_DONE: state <= ST_WAIT1;
                endcase
            end
        end
    end

endmodule


// =============================================================================
// Image RAM
// =============================================================================
module image_ram (
    input wire clk,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [9:0] rd_addr,
    output reg [7:0] rd_data
);
    (* ram_style = "block" *) reg [7:0] ram [0:783];
    
    always @(posedge clk) begin
        if (wr_en) ram[wr_addr] <= wr_data;
        rd_data <= ram[rd_addr];
    end
endmodule


// =============================================================================
// 7-Segment Display
// =============================================================================
module seven_segment_display (
    input wire clk,
    input wire rst,
    input wire [3:0] digit,
    output reg [6:0] seg,
    output reg [3:0] an
);
    always @(posedge clk) begin
        if (rst) begin
            an <= 4'b1110;
            seg <= 7'b1111111;
        end else begin
            an <= 4'b1110;
            case (digit)
                4'd0: seg <= 7'b1000000;
                4'd1: seg <= 7'b1111001;
                4'd2: seg <= 7'b0100100;
                4'd3: seg <= 7'b0110000;
                4'd4: seg <= 7'b0011001;
                4'd5: seg <= 7'b0010010;
                4'd6: seg <= 7'b0000010;
                4'd7: seg <= 7'b1111000;
                4'd8: seg <= 7'b0000000;
                4'd9: seg <= 7'b0010000;
                default: seg <= 7'b0111111;
            endcase
        end
    end
endmodule
