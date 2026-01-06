`timescale 1ns / 1ps

/*
================================================================================
Comprehensive CNN Pipeline Diagnostic Testbench
================================================================================
This testbench systematically checks:
  1. That the new code is actually being compiled (spy signals)
  2. Pipeline timing for BRAM reads
  3. Accumulator values at each step
  4. Comparison with bit-exact Python reference
================================================================================
*/

module tb_pipeline_diagnostic;

    // =========================================================================
    // DUT Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    wire done;
    wire [3:0] predicted_digit;
    
    wire [9:0] img_addr;
    wire [7:0] img_data;
    wire [12:0] conv_w_addr;
    wire [7:0] conv_w_data;
    wire [5:0] conv_b_addr;
    wire [31:0] conv_b_data;
    wire [12:0] dense_w_addr;
    wire [7:0] dense_w_data;
    wire [3:0] dense_b_addr;
    wire [31:0] dense_b_data;
    wire [13:0] buf_a_addr;
    wire [7:0] buf_a_wr_data;
    wire buf_a_wr_en;
    wire [7:0] buf_a_rd_data;
    wire [11:0] buf_b_addr;
    wire [7:0] buf_b_wr_data;
    wire buf_b_wr_en;
    wire [7:0] buf_b_rd_data;
    
    wire signed [31:0] class_score_0, class_score_1, class_score_2, class_score_3, class_score_4;
    wire signed [31:0] class_score_5, class_score_6, class_score_7, class_score_8, class_score_9;
    
    // =========================================================================
    // RAM Instances (Behavioral)
    // =========================================================================
    reg [7:0] image_ram [0:783];
    reg [7:0] conv_weights_ram [0:8191];
    reg [31:0] conv_biases_ram [0:47];
    reg [7:0] dense_weights_ram [0:7999];
    reg [31:0] dense_biases_ram [0:9];
    reg [7:0] buf_a_ram [0:10815];
    reg [7:0] buf_b_ram [0:2703];
    
    // Synchronous BRAM reads (1 cycle latency)
    reg [7:0] img_data_reg;
    reg [7:0] conv_w_data_reg;
    reg [31:0] conv_b_data_reg;
    reg [7:0] dense_w_data_reg;
    reg [31:0] dense_b_data_reg;
    reg [7:0] buf_a_rd_data_reg;
    reg [7:0] buf_b_rd_data_reg;
    
    always @(posedge clk) begin
        img_data_reg <= image_ram[img_addr];
        conv_w_data_reg <= conv_weights_ram[conv_w_addr];
        conv_b_data_reg <= conv_biases_ram[conv_b_addr];
        dense_w_data_reg <= dense_weights_ram[dense_w_addr];
        dense_b_data_reg <= dense_biases_ram[dense_b_addr];
        buf_a_rd_data_reg <= buf_a_ram[buf_a_addr];
        buf_b_rd_data_reg <= buf_b_ram[buf_b_addr];
    end
    
    assign img_data = img_data_reg;
    assign conv_w_data = conv_w_data_reg;
    assign conv_b_data = conv_b_data_reg;
    assign dense_w_data = dense_w_data_reg;
    assign dense_b_data = dense_b_data_reg;
    assign buf_a_rd_data = buf_a_rd_data_reg;
    assign buf_b_rd_data = buf_b_rd_data_reg;
    
    // Buffer writes
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    
    // =========================================================================
    // DUT Instantiation
    // =========================================================================
    inference dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .predicted_digit(predicted_digit),
        .img_addr(img_addr), .img_data(img_data),
        .conv_w_addr(conv_w_addr), .conv_w_data(conv_w_data),
        .conv_b_addr(conv_b_addr), .conv_b_data(conv_b_data),
        .dense_w_addr(dense_w_addr), .dense_w_data(dense_w_data),
        .dense_b_addr(dense_b_addr), .dense_b_data(dense_b_data),
        .buf_a_addr(buf_a_addr), .buf_a_wr_data(buf_a_wr_data),
        .buf_a_wr_en(buf_a_wr_en), .buf_a_rd_data(buf_a_rd_data),
        .buf_b_addr(buf_b_addr), .buf_b_wr_data(buf_b_wr_data),
        .buf_b_wr_en(buf_b_wr_en), .buf_b_rd_data(buf_b_rd_data),
        .class_score_0(class_score_0), .class_score_1(class_score_1),
        .class_score_2(class_score_2), .class_score_3(class_score_3),
        .class_score_4(class_score_4), .class_score_5(class_score_5),
        .class_score_6(class_score_6), .class_score_7(class_score_7),
        .class_score_8(class_score_8), .class_score_9(class_score_9)
    );
    
    // =========================================================================
    // Clock Generation (100 MHz)
    // =========================================================================
    always #5 clk = ~clk;
    
    // =========================================================================
    // Test Variables
    // =========================================================================
    integer i, j;
    reg [4:0] prev_state;
    integer cycle_count;
    
    // Golden reference for first conv output (calculated manually)
    reg signed [31:0] expected_acc;
    reg signed [7:0] pixel_val;
    reg signed [7:0] weight_val;
    
    // =========================================================================
    // State Name Decoder (for debug output)
    // =========================================================================
    function [127:0] state_name;
        input [4:0] state_val;
        begin
            case (state_val)
                5'd0:  state_name = "IDLE";
                5'd1:  state_name = "L1_LOAD_BIAS";
                5'd2:  state_name = "L1_LOAD_BIAS_WAIT";
                5'd3:  state_name = "L1_PREFETCH";
                5'd4:  state_name = "L1_CONV";
                5'd5:  state_name = "L1_SAVE";
                5'd6:  state_name = "L1_POOL";
                5'd7:  state_name = "L2_LOAD_BIAS";
                5'd8:  state_name = "L2_LOAD_BIAS_WAIT";
                5'd9:  state_name = "L2_PREFETCH";
                5'd10: state_name = "L2_CONV";
                5'd11: state_name = "L2_SAVE";
                5'd12: state_name = "L2_POOL";
                5'd13: state_name = "DENSE_LOAD_BIAS";
                5'd14: state_name = "DENSE_LOAD_BIAS_WAIT";
                5'd15: state_name = "DENSE_PREFETCH";
                5'd16: state_name = "DENSE_MULT";
                5'd17: state_name = "DENSE_NEXT";
                5'd18: state_name = "DONE_STATE";
                default: state_name = "UNKNOWN";
            endcase
        end
    endfunction
    
    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        prev_state = 5'd31; // Invalid state to force first print
        cycle_count = 0;
        
        $display("\n");
        $display("================================================================================");
        $display(" CNN PIPELINE DIAGNOSTIC TESTBENCH");
        $display("================================================================================");
        $display(" This test verifies:");
        $display("   1. New code is being compiled (state machine changes detected)");
        $display("   2. Pipeline timing is correct for BRAM reads");
        $display("   3. Accumulator values match expected");
        $display("================================================================================\n");
        
        // =====================================================================
        // STEP 1: Load Memory Files
        // =====================================================================
        $display("[SETUP] Loading memory files...");
        
        // Try to load real files, fall back to synthetic data
        if (!$fopen("sim_conv_weights.mem", "r")) begin
            $display("[SETUP] Memory files not found - using synthetic test data");
            
            // Create simple synthetic weights for testing
            // Conv1: 16 filters x 9 weights = 144 weights
            for (i = 0; i < 144; i = i + 1) begin
                conv_weights_ram[i] = (i % 5) - 2; // Values: -2, -1, 0, 1, 2, -2, ...
            end
            
            // Conv2: 32 filters x 16 channels x 9 = 4608 weights  
            for (i = 0; i < 4608; i = i + 1) begin
                conv_weights_ram[144 + i] = (i % 7) - 3;
            end
            
            // Conv1 biases: 16 x 32-bit
            for (i = 0; i < 16; i = i + 1) begin
                conv_biases_ram[i] = 32'h00000100; // Small positive bias (256)
            end
            
            // Conv2 biases: 32 x 32-bit
            for (i = 0; i < 32; i = i + 1) begin
                conv_biases_ram[16 + i] = 32'h00000080; // Small positive bias (128)
            end
            
            // Dense weights: 10 x 800 = 8000
            for (i = 0; i < 8000; i = i + 1) begin
                dense_weights_ram[i] = (i % 11) - 5;
            end
            
            // Dense biases: 10 x 32-bit
            for (i = 0; i < 10; i = i + 1) begin
                dense_biases_ram[i] = 32'h00001000; // 4096
            end
            
            // Create a simple test image (gradient pattern)
            for (i = 0; i < 784; i = i + 1) begin
                image_ram[i] = (i % 256); // 0-255 gradient
            end
            
        end else begin
            $readmemh("sim_conv_weights.mem", conv_weights_ram);
            $readmemh("sim_conv_biases.mem", conv_biases_ram);
            $readmemh("sim_dense_weights.mem", dense_weights_ram);
            $readmemh("sim_dense_biases.mem", dense_biases_ram);
            $readmemh("test_pixels.mem", image_ram, 0, 783);
            $display("[SETUP] Real memory files loaded successfully");
        end
        
        // =====================================================================
        // STEP 2: Calculate Expected Values Manually
        // =====================================================================
        $display("\n[CALC] Calculating expected first convolution output...");
        
        // First conv output at (r=0, c=0, f=0)
        expected_acc = $signed(conv_biases_ram[0]);
        $display("  Bias[0] = %0d (0x%08h)", $signed(conv_biases_ram[0]), conv_biases_ram[0]);
        
        for (i = 0; i < 3; i = i + 1) begin
            for (j = 0; j < 3; j = j + 1) begin
                pixel_val = $signed(image_ram[i * 28 + j]);
                weight_val = $signed(conv_weights_ram[i * 3 + j]);
                $display("  Pixel[%0d,%0d]=%4d * Weight[%0d]=%4d = %6d", 
                    i, j, pixel_val, i*3+j, weight_val, pixel_val * weight_val);
                expected_acc = expected_acc + (pixel_val * weight_val);
            end
        end
        
        $display("  Raw Accumulator = %0d", expected_acc);
        $display("  After Shift (>>>8) = %0d", expected_acc >>> 8);
        if ((expected_acc >>> 8) < 0)
            $display("  After ReLU = 0");
        else if ((expected_acc >>> 8) > 127)
            $display("  After ReLU+Sat = 127");
        else
            $display("  After ReLU+Sat = %0d", expected_acc >>> 8);
        
        // =====================================================================
        // STEP 3: Reset and Start
        // =====================================================================
        $display("\n[RUN] Starting simulation...");
        #100;
        rst = 0;
        #100;
        
        // Pulse start signal properly
        @(posedge clk);
        #1;
        start = 1;
        @(posedge clk);
        #1;
        start = 0;
        
        // =====================================================================
        // STEP 4: Monitor State Transitions
        // =====================================================================
        $display("\n[TRACE] State machine transitions:");
        $display("--------------------------------------------------------------------------------");
        
        while (!done && cycle_count < 10000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
            
            // Print on state change
            if (dut.state !== prev_state) begin
                $display("  Cycle %5d: %s -> %s", 
                    cycle_count, state_name(prev_state), state_name(dut.state));
                
                // Special monitoring for L1 first convolution
                if (dut.state == 5'd3 && dut.f_idx == 0 && dut.r == 0 && dut.c == 0) begin
                    $display("  *** ENTERED L1_PREFETCH - NEW CODE IS ACTIVE! ***");
                end
                
                if (dut.state == 5'd4 && dut.f_idx == 0 && dut.r == 0 && dut.c == 0) begin
                    $display("  *** L1_CONV: Accumulator = %0d ***", dut.acc);
                end
                
                if (dut.state == 5'd5 && dut.f_idx == 0 && dut.r == 0 && dut.c == 0) begin
                    $display("  *** L1_SAVE: Final Acc = %0d, Output = %0d ***", 
                        dut.acc, buf_a_wr_data);
                    $display("      Expected Acc = %0d", expected_acc);
                    if (dut.acc === expected_acc)
                        $display("      [PASS] Accumulator matches expected!");
                    else
                        $display("      [FAIL] Accumulator mismatch! Delta = %0d", 
                            dut.acc - expected_acc);
                end
                
                prev_state = dut.state;
            end
            
            // Limit output for long runs
            if (cycle_count == 1000) begin
                $display("\n  ... (skipping to completion) ...\n");
            end
        end
        
        // =====================================================================
        // STEP 5: Final Results
        // =====================================================================
        $display("\n================================================================================");
        $display(" FINAL RESULTS");
        $display("================================================================================");
        
        if (done) begin
            $display("  Inference completed in %0d cycles", cycle_count);
            $display("  Predicted digit: %0d", predicted_digit);
            $display("\n  Class Scores:");
            $display("    [0] = %12d", class_score_0);
            $display("    [1] = %12d", class_score_1);
            $display("    [2] = %12d", class_score_2);
            $display("    [3] = %12d", class_score_3);
            $display("    [4] = %12d", class_score_4);
            $display("    [5] = %12d", class_score_5);
            $display("    [6] = %12d", class_score_6);
            $display("    [7] = %12d", class_score_7);
            $display("    [8] = %12d", class_score_8);
            $display("    [9] = %12d", class_score_9);
        end else begin
            $display("  [ERROR] Inference did not complete within timeout!");
            $display("  Final state: %s (state=%0d)", state_name(dut.state), dut.state);
        end
        
        $display("\n================================================================================\n");
        $stop;
    end
    
    // =========================================================================
    // Detailed L1 CONV Trace (First Output Only)
    // =========================================================================
    always @(posedge clk) begin
        // Only trace first convolution (f_idx=0, r=0, c=0)
        if (dut.state == 5'd4 && dut.f_idx == 0 && dut.r == 0 && dut.c == 0) begin
            $display("    L1_CONV: kr=%0d kc=%0d | img_addr=%0d img_data=%0d | w_addr=%0d w_data=%0d | acc=%0d",
                dut.kr, dut.kc, img_addr, $signed(img_data), 
                conv_w_addr, $signed(conv_w_data), dut.acc);
        end
    end
    
    // =========================================================================
    // Timeout Watchdog
    // =========================================================================
    initial begin
        #50_000_000; // 50ms timeout
        $display("\n[ERROR] Simulation timeout!");
        $stop;
    end

endmodule