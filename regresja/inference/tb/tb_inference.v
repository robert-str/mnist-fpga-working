/*
================================================================================
Testbench for Inference Module
================================================================================
This testbench verifies the core inference logic without requiring the FPGA board.
It mocks the memory interfaces and tests the computation.
================================================================================
*/

`timescale 1ns / 1ps

module tb_inference();

    reg clk;
    reg rst;
    
    // Mocks for Memory Interfaces
    wire [12:0] weight_addr;
    reg  [7:0]  weight_data;
    wire [3:0]  bias_addr;
    reg  [31:0] bias_data;
    wire [9:0]  input_addr;
    reg  [7:0]  input_pixel;
    
    // Control
    reg weights_ready;
    reg start_inference;
    
    // Outputs
    wire [3:0] predicted_digit;
    wire inference_done;
    wire busy;

    // Instantiate the Unit Under Test (UUT)
    inference uut (
        .clk(clk),
        .rst(rst),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .bias_addr(bias_addr),
        .bias_data(bias_data),
        .weights_ready(weights_ready),
        .start_inference(start_inference),
        .input_pixel(input_pixel),
        .input_addr(input_addr),
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(busy)
    );

    // Clock Generation
    always #5 clk = ~clk; // 100MHz (10ns period)

    // --- Mock Memory Logic ---
    // Simple test: Let's make Digit 2 the winner.
    // We will make weights for Class 2 positive, others zero/negative.
    always @(posedge clk) begin
        // Weight Memory Mock
        // Addr logic: Class = weight_addr / 784
        if ((weight_addr / 784) == 2) 
            weight_data <= 8'd10; // Positive weight for class 2
        else 
            weight_data <= -8'd10; // Negative for others
            
        // Bias Memory Mock
        bias_data <= 32'd0; 
        
        // Input Image Mock (All white pixels)
        input_pixel <= 8'd10; 
    end

    initial begin
        // Initialize Inputs
        clk = 0;
        rst = 1;
        weights_ready = 0;
        start_inference = 0;
        weight_data = 0;
        bias_data = 0;
        input_pixel = 0;

        // Reset Pulse
        #100;
        rst = 0;
        #20;

        // Start Test
        $display("Starting Inference Simulation...");
        $display("Time: %t", $time);
        weights_ready = 1;
        
        // Pulse start
        #10 start_inference = 1;
        #10 start_inference = 0;

        // Wait for done
        wait(inference_done);
        
        // Check Result
        $display("Inference Done at time %t", $time);
        $display("Predicted Digit: %d", predicted_digit);
        
        if (predicted_digit == 4'd2) 
            $display("TEST PASSED: Correctly predicted 2 based on mock weights.");
        else 
            $display("TEST FAILED: Expected 2, got %d", predicted_digit);

        #100;
        $finish;
    end

endmodule
