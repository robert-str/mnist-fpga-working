`timescale 1ns / 1ps

/*
================================================================================
First Convolution Deep Trace
================================================================================
Traces every single accumulation of the first L1 convolution (filter 0, r=0, c=0)
to find exactly where FPGA diverges from expected.
================================================================================
*/

module tb_first_conv_trace;

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
    
    // RAMs - COMBINATIONAL reads (matches your tb_inference.v)
    reg [7:0] image_ram [0:783];
    reg [7:0] conv_weights_ram [0:8191];
    reg [31:0] conv_biases_ram [0:47];
    reg [7:0] dense_weights_ram [0:7999];
    reg [31:0] dense_biases_ram [0:9];
    reg [7:0] buf_a_ram [0:10815];
    reg [7:0] buf_b_ram [0:2703];
    
    // COMBINATIONAL reads
    assign img_data = image_ram[img_addr];
    assign conv_w_data = conv_weights_ram[conv_w_addr];
    assign conv_b_data = conv_biases_ram[conv_b_addr];
    assign dense_w_data = dense_weights_ram[dense_w_addr];
    assign dense_b_data = dense_biases_ram[dense_b_addr];
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];
    
    // Buffer writes - synchronous
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    
    // DUT
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
    
    // Manual calculation
    reg signed [31:0] manual_acc;
    reg signed [7:0] px, wt;
    integer i, j;
    
    initial begin
        clk = 0;
        rst = 1;
        start = 0;
        
        $display("\n================================================================================");
        $display(" FIRST CONVOLUTION DEEP TRACE");
        $display("================================================================================\n");
        
        // Load memories
        $readmemh("sim_conv_weights.mem", conv_weights_ram);
        $readmemh("sim_conv_biases.mem", conv_biases_ram);
        $readmemh("sim_dense_weights.mem", dense_weights_ram);
        $readmemh("sim_dense_biases.mem", dense_biases_ram);
        $readmemh("test_pixels.mem", image_ram, 0, 783);
        
        // =====================================================================
        // Manual calculation for first output (f=0, r=0, c=0)
        // =====================================================================
        $display("[MANUAL] Calculating expected first conv output...\n");
        
        manual_acc = $signed(conv_biases_ram[0]);
        $display("  Bias = %0d (0x%08h)", $signed(conv_biases_ram[0]), conv_biases_ram[0]);
        $display("");
        $display("  3x3 Kernel Calculation:");
        $display("  kr | kc | img_addr | pixel | w_addr | weight |  product  | running_acc");
        $display("  ---|----| ---------|-------|--------|--------|-----------|------------");
        
        for (i = 0; i < 3; i = i + 1) begin
            for (j = 0; j < 3; j = j + 1) begin
                px = $signed(image_ram[i * 28 + j]);
                wt = $signed(conv_weights_ram[i * 3 + j]);
                $display("   %0d |  %0d |    %4d  |  %4d |   %4d |   %4d |  %8d | %11d",
                    i, j, i * 28 + j, px, i * 3 + j, wt, px * wt, manual_acc + px * wt);
                manual_acc = manual_acc + (px * wt);
            end
        end
        
        $display("");
        $display("  Final Accumulator: %0d", manual_acc);
        $display("  After Shift (>>>8): %0d", manual_acc >>> 8);
        if ((manual_acc >>> 8) < 0)
            $display("  After ReLU: 0");
        else if ((manual_acc >>> 8) > 127)
            $display("  After ReLU+Sat: 127");
        else
            $display("  After ReLU+Sat: %0d", manual_acc >>> 8);
        
        // =====================================================================
        // Run FPGA and trace
        // =====================================================================
        $display("\n[FPGA] Running inference and tracing first conv...\n");
        
        #100;
        rst = 0;
        #100;
        
        @(posedge clk);
        #1;
        start = 1;
        @(posedge clk);
        #1;
        start = 0;
        
        // Wait for first write to buf_a
        wait(buf_a_wr_en && buf_a_addr == 0);
        @(posedge clk);
        
        $display("  FPGA First Output:");
        $display("    Accumulator (dut.acc): %0d", dut.acc);
        $display("    Written Value: %0d", buf_a_wr_data);
        $display("    Address: %0d", buf_a_addr);
        
        if (manual_acc >>> 8 == buf_a_wr_data || 
            ((manual_acc >>> 8) < 0 && buf_a_wr_data == 0) ||
            ((manual_acc >>> 8) > 127 && buf_a_wr_data == 127))
            $display("\n  [MATCH] First conv output matches manual calculation!");
        else
            $display("\n  [MISMATCH] Expected %0d, got %0d", 
                ((manual_acc >>> 8) < 0) ? 0 : (((manual_acc >>> 8) > 127) ? 127 : (manual_acc >>> 8)),
                buf_a_wr_data);
        
        // Continue to completion
        wait(done);
        @(posedge clk);
        
        $display("\n[DONE] Inference complete.");
        $display("  Predicted digit: %0d", predicted_digit);
        $display("  Score[7]: %0d", class_score_7);
        
        $display("\n================================================================================\n");
        $stop;
    end
    
    // Trace every state transition for first conv
    reg [4:0] prev_state;
    reg first_conv_done;
    
    initial begin
        prev_state = 5'd31;
        first_conv_done = 0;
    end
    
    always @(posedge clk) begin
        if (!first_conv_done && dut.f_idx == 0 && dut.r == 0 && dut.c == 0) begin
            if (dut.state != prev_state) begin
                $display("  [STATE %0d] img_addr=%0d img_data=%0d | w_addr=%0d w_data=%0d | acc=%0d",
                    dut.state, img_addr, $signed(img_data), conv_w_addr, $signed(conv_w_data), dut.acc);
                prev_state <= dut.state;
            end
            
            // Mark done when we leave first conv output
            if (buf_a_wr_en && buf_a_addr == 0) begin
                first_conv_done <= 1;
            end
        end
    end
    
    // Timeout
    initial begin
        #50_000_000;
        $display("\n[ERROR] Timeout!");
        $stop;
    end

endmodule