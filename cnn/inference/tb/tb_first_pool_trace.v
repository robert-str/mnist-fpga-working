`timescale 1ns / 1ps

/*
================================================================================
L1 Pool Diagnostic - Check if pooling is correct
================================================================================
Since L1 Conv matches, let's verify L1 Pool.
================================================================================
*/

module tb_l1_pool_diagnostic;

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
    
    // RAMs - COMBINATIONAL reads
    reg [7:0] image_ram [0:783];
    reg [7:0] conv_weights_ram [0:8191];
    reg [31:0] conv_biases_ram [0:47];
    reg [7:0] dense_weights_ram [0:7999];
    reg [31:0] dense_biases_ram [0:9];
    reg [7:0] buf_a_ram [0:10815];
    reg [7:0] buf_b_ram [0:2703];
    
    assign img_data = image_ram[img_addr];
    assign conv_w_data = conv_weights_ram[conv_w_addr];
    assign conv_b_data = conv_biases_ram[conv_b_addr];
    assign dense_w_data = dense_weights_ram[dense_w_addr];
    assign dense_b_data = dense_biases_ram[dense_b_addr];
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];
    
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    
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
    
    always #5 clk = ~clk;
    
    // =========================================================================
    // Python-equivalent calculations
    // =========================================================================
    reg [7:0] py_l1_conv [0:10815];
    reg [7:0] py_l1_pool [0:2703];
    
    task compute_py_l1_conv;
        integer f, row, col, kr, kc;
        reg signed [31:0] acc;
        reg signed [7:0] px, wt;
        reg signed [31:0] shifted;
        begin
            for (f = 0; f < 16; f = f + 1) begin
                for (row = 0; row < 26; row = row + 1) begin
                    for (col = 0; col < 26; col = col + 1) begin
                        acc = $signed(conv_biases_ram[f]);
                        for (kr = 0; kr < 3; kr = kr + 1) begin
                            for (kc = 0; kc < 3; kc = kc + 1) begin
                                px = $signed(image_ram[(row + kr) * 28 + (col + kc)]);
                                wt = $signed(conv_weights_ram[f * 9 + kr * 3 + kc]);
                                acc = acc + px * wt;
                            end
                        end
                        shifted = acc >>> 8;
                        if (shifted < 0) shifted = 0;
                        if (shifted > 127) shifted = 127;
                        py_l1_conv[f * 676 + row * 26 + col] = shifted[7:0];
                    end
                end
            end
        end
    endtask
    
    task compute_py_l1_pool;
        integer f, row, col;
        reg [7:0] v0, v1, v2, v3, mx;
        begin
            for (f = 0; f < 16; f = f + 1) begin
                for (row = 0; row < 13; row = row + 1) begin
                    for (col = 0; col < 13; col = col + 1) begin
                        v0 = py_l1_conv[f * 676 + (row*2) * 26 + (col*2)];
                        v1 = py_l1_conv[f * 676 + (row*2) * 26 + (col*2+1)];
                        v2 = py_l1_conv[f * 676 + (row*2+1) * 26 + (col*2)];
                        v3 = py_l1_conv[f * 676 + (row*2+1) * 26 + (col*2+1)];
                        mx = v0;
                        if (v1 > mx) mx = v1;
                        if (v2 > mx) mx = v2;
                        if (v3 > mx) mx = v3;
                        py_l1_pool[f * 169 + row * 13 + col] = mx;
                    end
                end
            end
        end
    endtask
    
    // =========================================================================
    // Main test
    // =========================================================================
    integer i, mismatches;
    reg l1_pool_started;
    reg l2_started;
    
    initial begin
        clk = 0;
        rst = 1;
        start = 0;
        l1_pool_started = 0;
        l2_started = 0;
        
        $display("\n================================================================================");
        $display(" L1 POOL DIAGNOSTIC");
        $display("================================================================================\n");
        
        // Load
        $readmemh("sim_conv_weights.mem", conv_weights_ram);
        $readmemh("sim_conv_biases.mem", conv_biases_ram);
        $readmemh("sim_dense_weights.mem", dense_weights_ram);
        $readmemh("sim_dense_biases.mem", dense_biases_ram);
        $readmemh("test_pixels.mem", image_ram, 0, 783);
        
        // Compute Python reference
        $display("[PYTHON] Computing L1 Conv reference...");
        compute_py_l1_conv();
        $display("[PYTHON] Computing L1 Pool reference...");
        compute_py_l1_pool();
        
        // Show first few Python L1 conv outputs
        $display("\n[PYTHON] First 5 L1 Conv outputs (filter 0):");
        for (i = 0; i < 5; i = i + 1) begin
            $display("  py_l1_conv[%0d] = %0d", i, py_l1_conv[i]);
        end
        
        // Show first Python L1 pool output and its inputs
        $display("\n[PYTHON] First L1 Pool calculation (f=0, r=0, c=0):");
        $display("  Inputs from L1 Conv:");
        $display("    [0,0] = %0d", py_l1_conv[0]);
        $display("    [0,1] = %0d", py_l1_conv[1]);
        $display("    [1,0] = %0d", py_l1_conv[26]);
        $display("    [1,1] = %0d", py_l1_conv[27]);
        $display("  Max = %0d", py_l1_pool[0]);
        
        // Run FPGA
        $display("\n[FPGA] Running inference...");
        #100;
        rst = 0;
        #100;
        
        @(posedge clk);
        #1;
        start = 1;
        @(posedge clk);
        #1;
        start = 0;
        
        // Wait for L2 to start (means L1 pool is done)
        wait(dut.state == 7);  // L2_LOAD_BIAS
        @(posedge clk);
        @(posedge clk);
        
        // Compare L1 Conv outputs
        $display("\n[CHECK] Comparing L1 Conv outputs...");
        mismatches = 0;
        for (i = 0; i < 10816; i = i + 1) begin
            if (buf_a_ram[i] !== py_l1_conv[i]) begin
                if (mismatches < 10) begin
                    $display("  MISMATCH at [%0d]: FPGA=%0d, Python=%0d", i, buf_a_ram[i], py_l1_conv[i]);
                end
                mismatches = mismatches + 1;
            end
        end
        if (mismatches == 0)
            $display("  [PASS] All 10816 L1 Conv values match!");
        else
            $display("  [FAIL] %0d mismatches in L1 Conv", mismatches);
        
        // Compare L1 Pool outputs
        $display("\n[CHECK] Comparing L1 Pool outputs...");
        mismatches = 0;
        for (i = 0; i < 2704; i = i + 1) begin
            if (buf_b_ram[i] !== py_l1_pool[i]) begin
                if (mismatches < 10) begin
                    $display("  MISMATCH at [%0d]: FPGA=%0d, Python=%0d", i, buf_b_ram[i], py_l1_pool[i]);
                end
                mismatches = mismatches + 1;
            end
        end
        if (mismatches == 0)
            $display("  [PASS] All 2704 L1 Pool values match!");
        else
            $display("  [FAIL] %0d mismatches in L1 Pool", mismatches);
        
        // Show first few FPGA L1 pool outputs
        $display("\n[FPGA] First 5 L1 Pool outputs:");
        for (i = 0; i < 5; i = i + 1) begin
            $display("  buf_b_ram[%0d] = %0d (expected %0d) %s", 
                i, buf_b_ram[i], py_l1_pool[i],
                (buf_b_ram[i] == py_l1_pool[i]) ? "" : "<-- MISMATCH");
        end
        
        // Continue to completion
        wait(done);
        @(posedge clk);
        
        $display("\n[DONE] Final scores:");
        $display("  Score[7] = %0d", class_score_7);
        
        $display("\n================================================================================\n");
        $stop;
    end
    
    // Timeout
    initial begin
        #100_000_000;
        $display("\n[ERROR] Timeout!");
        $stop;
    end

endmodule