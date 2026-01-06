`timescale 1ns / 1ps

/*
================================================================================
Enhanced CNN Inference Testbench - Systematic Golden Reference Comparison
================================================================================
Tests 100 MNIST images and compares:
  1. Predicted digit (FPGA vs Python)
  2. All 10 class scores/logits (FPGA vs Python)
  3. Reports mismatches and statistics
================================================================================
*/

module tb_inference_cnn;

    // =========================================================================
    // Constants
    // =========================================================================
    localparam NUM_TESTS = 100; // <--- INCREASED TO 100
    localparam IMAGE_SIZE = 784;
    localparam NUM_CLASSES = 10;
    
    // =========================================================================
    // DUT Signals
    // =========================================================================
    reg clk;
    reg rst;
    reg start;
    wire done;
    wire [3:0] predicted_digit;
    
    // Image RAM
    wire [9:0] img_addr;
    wire [7:0] img_data;
    
    // Conv RAMs
    wire [12:0] conv_w_addr;
    wire [7:0] conv_w_data;
    wire [5:0] conv_b_addr;
    wire [31:0] conv_b_data;
    
    // Dense RAMs
    wire [12:0] dense_w_addr;
    wire [7:0] dense_w_data;
    wire [3:0] dense_b_addr;
    wire [31:0] dense_b_data;
    
    // Buffers
    wire [13:0] buf_a_addr;
    wire [7:0] buf_a_wr_data;
    wire buf_a_wr_en;
    wire [7:0] buf_a_rd_data;
    
    wire [11:0] buf_b_addr;
    wire [7:0] buf_b_wr_data;
    wire buf_b_wr_en;
    wire [7:0] buf_b_rd_data;
    
    // Class Scores
    wire signed [31:0] class_score_0, class_score_1, class_score_2, class_score_3, class_score_4;
    wire signed [31:0] class_score_5, class_score_6, class_score_7, class_score_8, class_score_9;
    
    // =========================================================================
    // Golden Reference Storage
    // =========================================================================
    reg [7:0] golden_pixels [0:NUM_TESTS*IMAGE_SIZE-1];
    reg [31:0] golden_scores [0:NUM_TESTS*NUM_CLASSES-1];
    reg [3:0] golden_preds [0:NUM_TESTS-1];
    reg [3:0] golden_labels [0:NUM_TESTS-1];
    
    // =========================================================================
    // RAM Instances
    // =========================================================================
    
    // Image RAM (for loading test images)
    reg [7:0] image_ram [0:783];
    assign img_data = image_ram[img_addr];
    
    // Conv Weights RAM
    reg [7:0] conv_weights_ram [0:8191];
    assign conv_w_data = conv_weights_ram[conv_w_addr];
    
    // Conv Biases RAM
    reg [31:0] conv_biases_ram [0:47];
    assign conv_b_data = conv_biases_ram[conv_b_addr];
    
    // Dense Weights RAM (in ram_cnn module)
    reg [7:0] dense_weights_ram [0:7999];
    reg [7:0] buf_a_ram [0:10815];
    reg [7:0] buf_b_ram [0:2703];
    
    // Buffer A
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
    end
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    
    // Buffer B
    always @(posedge clk) begin
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];
    
    // Dense Weights (accessed by inference module)
    assign dense_w_data = dense_weights_ram[dense_w_addr];
    
    // Dense Biases RAM
    reg [31:0] dense_biases_ram [0:9];
    assign dense_b_data = dense_biases_ram[dense_b_addr];
    
    // =========================================================================
    // DUT Instantiation
    // =========================================================================
    inference dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .predicted_digit(predicted_digit),
        
        .img_addr(img_addr),
        .img_data(img_data),
        
        .conv_w_addr(conv_w_addr),
        .conv_w_data(conv_w_data),
        .conv_b_addr(conv_b_addr),
        .conv_b_data(conv_b_data),
        
        .dense_w_addr(dense_w_addr),
        .dense_w_data(dense_w_data),
        .dense_b_addr(dense_b_addr),
        .dense_b_data(dense_b_data),
        
        .buf_a_addr(buf_a_addr),
        .buf_a_wr_data(buf_a_wr_data),
        .buf_a_wr_en(buf_a_wr_en),
        .buf_a_rd_data(buf_a_rd_data),
        
        .buf_b_addr(buf_b_addr),
        .buf_b_wr_data(buf_b_wr_data),
        .buf_b_wr_en(buf_b_wr_en),
        .buf_b_rd_data(buf_b_rd_data),
        
        .class_score_0(class_score_0),
        .class_score_1(class_score_1),
        .class_score_2(class_score_2),
        .class_score_3(class_score_3),
        .class_score_4(class_score_4),
        .class_score_5(class_score_5),
        .class_score_6(class_score_6),
        .class_score_7(class_score_7),
        .class_score_8(class_score_8),
        .class_score_9(class_score_9)
    );
    
    // =========================================================================
    // Clock Generation (100 MHz)
    // =========================================================================
    always #5 clk = ~clk;
    
    // =========================================================================
    // Test Statistics
    // =========================================================================
    integer tests_passed;
    integer tests_failed;
    integer pred_correct;
    integer pred_incorrect;
    
    // =========================================================================
    // Helper Tasks
    // =========================================================================
    
    // Load a single image into image_ram
    task load_image;
        input integer img_idx;
        integer i;
        integer src_addr;
        begin
            src_addr = img_idx * IMAGE_SIZE;
            for (i = 0; i < IMAGE_SIZE; i = i + 1) begin
                image_ram[i] = golden_pixels[src_addr + i];
            end
            $display("[LOAD] Image %0d loaded into RAM", img_idx);
        end
    endtask
    
    // Compare FPGA scores with golden reference
    task compare_scores;
        input integer img_idx;
        reg signed [31:0] fpga_scores [0:9];
        reg signed [31:0] gold_scores [0:9];
        integer base_addr;
        integer i;
        integer mismatches;
        integer max_diff;
        integer diff;
        begin
            // Extract FPGA scores
            fpga_scores[0] = class_score_0;
            fpga_scores[1] = class_score_1;
            fpga_scores[2] = class_score_2;
            fpga_scores[3] = class_score_3;
            fpga_scores[4] = class_score_4;
            fpga_scores[5] = class_score_5;
            fpga_scores[6] = class_score_6;
            fpga_scores[7] = class_score_7;
            fpga_scores[8] = class_score_8;
            fpga_scores[9] = class_score_9;
            
            // Extract golden scores
            base_addr = img_idx * NUM_CLASSES;
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                gold_scores[i] = golden_scores[base_addr + i];
            end
            
            // Compare
            mismatches = 0;
            max_diff = 0;
            
            // Only output detailed scores on mismatch to save console log space for 100 runs
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                diff = fpga_scores[i] - gold_scores[i];
                if (diff < 0) diff = -diff;
                
                if (diff > max_diff) max_diff = diff;
                if (fpga_scores[i] !== gold_scores[i]) mismatches = mismatches + 1;
            end
            
            if (mismatches == 0) begin
                tests_passed = tests_passed + 1;
                $display("  [OK] Scores Match. Max Diff: %0d", max_diff);
            end else begin
                tests_failed = tests_failed + 1;
                $display("  [MM] %0d score mismatches detected", mismatches);
                $display("  Class |      FPGA      |     Golden     | Difference");
                $display("  ------|------------|------------|------------");
                for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                    $display("    %0d    | %10d | %10d | %10d %s",
                        i, fpga_scores[i], gold_scores[i], fpga_scores[i] - gold_scores[i],
                        (fpga_scores[i] === gold_scores[i]) ? "[OK]" : "[MM]");
                end
            end
        end
    endtask
    
    // Run inference on a single image
    task run_inference;
        input integer img_idx;
        reg [7:0] probe_pixel;
        begin
            $display("\n========================================");
            $display("Test %0d: Image Index %0d", img_idx + 1, img_idx);
            
            // 1. Load image
            load_image(img_idx);
            
            // Wait a bit for RAM to stabilize
            repeat(10) @(posedge clk);
            
            // 2. Start inference
            @(posedge clk);
            #1;            
            start = 1;     
            @(posedge clk);
            #1;
            start = 0;     
            
            // Wait for completion
            wait(done);
            @(posedge clk);
            
            // 3. Check prediction
            $display("  Expected: %0d | Predicted: %0d", golden_preds[img_idx], predicted_digit);
            
            if (predicted_digit === golden_preds[img_idx]) begin
                pred_correct = pred_correct + 1;
            end else begin
                $display("  [MM] PREDICTION MISMATCH!");
                pred_incorrect = pred_incorrect + 1;
            end
            
            // Compare scores
            compare_scores(img_idx);
            
            // Wait before next test
            repeat(10) @(posedge clk);
        end
    endtask
    
    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    integer i;
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        tests_passed = 0;
        tests_failed = 0;
        pred_correct = 0;
        pred_incorrect = 0;
        
        $display("\n================================================================================");
        $display(" CNN Inference Testbench - 100 Image Run");
        $display("================================================================================\n");
        
        // Load memory files
        $display("Loading memory files...");
        
        if ($feof($fopen("sim_conv_weights.mem", "r"))) begin
            $display("ERROR: sim_conv_weights.mem not found!");
            $display("Run generate_cnn_vectors.py first!");
            $stop;
        end
        
        $readmemh("sim_conv_weights.mem", conv_weights_ram);
        $readmemh("sim_conv_biases.mem", conv_biases_ram);
        $readmemh("sim_dense_weights.mem", dense_weights_ram);
        $readmemh("sim_dense_biases.mem", dense_biases_ram);
        
        $readmemh("test_pixels.mem", golden_pixels);
        $readmemh("test_scores.mem", golden_scores);
        $readmemh("test_preds.mem", golden_preds);
        $readmemh("test_labels.mem", golden_labels);
        
        $display("  > Weights loaded");
        $display("  > Test vectors loaded (100 images)\n");
        
        // Reset
        #100;
        rst = 0;
        #100;
        
        // Run tests on all images
        for (i = 0; i < NUM_TESTS; i = i + 1) begin
            run_inference(i);
        end
        
        // Final Summary
        $display("\n\n================================================================================");
        $display(" FINAL RESULTS (100 IMAGES)");
        $display("================================================================================");
        $display("\nPrediction Accuracy:");
        $display("  Correct:   %0d/%0d (%.1f%%)", pred_correct, NUM_TESTS, 
            (pred_correct * 100.0) / NUM_TESTS);
        $display("  Incorrect: %0d/%0d", pred_incorrect, NUM_TESTS);
        
        $display("\nScore Comparison:");
        $display("  Exact Matches: %0d/%0d tests", tests_passed, NUM_TESTS);
        $display("  Mismatches:    %0d/%0d tests", tests_failed, NUM_TESTS);
        
        if (tests_passed === NUM_TESTS) begin
            $display("\n[SUCCESS] All 100 tests passed! FPGA matches Python exactly.");
        end else begin
            $display("\n[FAILURE] Some tests failed.");
        end
        
        $display("\n================================================================================\n");
        $stop;
    end
    
    // =========================================================================
    // Timeout Watchdog - INCREASED FOR 100 IMAGES
    // =========================================================================
    initial begin
        #1_500_000_000; // 1.5 Billion ns timeout
        $display("\n[ERROR] Simulation timeout!");
        $display("Inference loop took too long or stalled.");
        $stop;
    end

endmodule