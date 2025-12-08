/*
================================================================================
Top Module - Complete System with Weight Loading and Inference
================================================================================
This module instantiates all sub-modules and connects them together.
Replaces the original softmax_regression_top module.
================================================================================
*/

module top (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    output wire tx,               // UART TX Line (optional, for debug)
    output wire [15:0] led,       // Status LEDs
    output wire [6:0] seg,        // 7-segment display segments
    output wire [3:0] an,         // 7-segment display anodes
    input wire [15:0] sw,         // Switches (for control/debug)
    input wire btnU,              // Up button (optional)
    input wire btnD,              // Down button (optional)
    input wire btnL,              // Left button (optional)
    input wire btnR               // Right button (optional)
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    
    // Weight loader signals
    wire [12:0] weight_rd_addr;
    wire [7:0]  weight_rd_data;
    wire [3:0]  bias_rd_addr;
    wire [31:0] bias_rd_data;
    wire        weights_loaded;
    wire [15:0] loader_led;
    
    // Inference signals
    wire [12:0] inf_weight_addr;
    wire [3:0]  inf_bias_addr;
    wire [9:0]  inf_input_addr;
    wire [7:0]  inf_input_pixel;
    wire [3:0]  predicted_digit;
    wire        inference_done;
    wire        inference_busy;
    
    // Image RAM signals
    wire [9:0]  img_wr_addr;
    wire [7:0]  img_wr_data;
    wire        img_wr_en;
    wire        img_loaded;
    
    // Control signals
    wire start_inference_pulse;
    
    // Predicted digit RAM signals
    wire digit_ram_wr_en;
    wire [7:0] digit_ram_rd_data;
    
    // UART TX signals for digit reader
    wire [7:0] tx_data;
    wire tx_send;
    wire tx_busy;
    wire tx_out;
    
    // UART RX signals for digit reader
    wire [7:0] digit_rx_data;
    wire digit_rx_ready;
    
    // LED and display registers
    reg [15:0] led_reg;
    
    // Display digit signals
    wire [3:0] digit_left;   // From memory (persistent)
    wire [3:0] digit_right;  // Current predicted_digit
    
    // Edge detector for inference_done (to create write pulse)
    reg inference_done_prev;
    
    // =========================================================================
    // Weight Loader
    // =========================================================================
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weight_rd_addr(weight_rd_addr),
        .weight_rd_data(weight_rd_data),
        .bias_rd_addr(bias_rd_addr),
        .bias_rd_data(bias_rd_data),
        .transfer_done(weights_loaded),
        .led(loader_led)
    );
    
    // =========================================================================
    // Image Loader (separate UART protocol)
    // =========================================================================
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
    
    // =========================================================================
    // Image RAM (784 bytes)
    // =========================================================================
    image_ram u_image_ram (
        .clk(clk),
        .wr_addr(img_wr_addr),
        .wr_data(img_wr_data),
        .wr_en(img_wr_en),
        .rd_addr(inf_input_addr),
        .rd_data(inf_input_pixel)
    );
    
    // =========================================================================
    // Inference Module
    // =========================================================================
    
    // Connect weight/bias addresses from inference to loader
    assign weight_rd_addr = inf_weight_addr;
    assign bias_rd_addr = inf_bias_addr;
    
    inference u_inference (
        .clk(clk),
        .rst(rst),
        .weight_addr(inf_weight_addr),
        .weight_data(weight_rd_data),
        .bias_addr(inf_bias_addr),
        .bias_data(bias_rd_data),
        .weights_ready(weights_loaded),
        .start_inference(start_inference_pulse),
        .input_pixel(inf_input_pixel),
        .input_addr(inf_input_addr),
        .predicted_digit(predicted_digit),
        .inference_done(inference_done),
        .busy(inference_busy)
    );
    
    // =========================================================================
    // Auto-start inference when image is loaded
    // =========================================================================
    reg img_loaded_prev;
    always @(posedge clk) begin
        if (rst)
            img_loaded_prev <= 0;
        else
            img_loaded_prev <= img_loaded;
    end
    
    assign start_inference_pulse = img_loaded && !img_loaded_prev;
    
    // =========================================================================
    // Edge detector for inference_done (to write predicted digit to RAM)
    // =========================================================================
    always @(posedge clk) begin
        if (rst)
            inference_done_prev <= 0;
        else
            inference_done_prev <= inference_done;
    end
    
    assign digit_ram_wr_en = inference_done && !inference_done_prev;
    
    // =========================================================================
    // Predicted Digit RAM
    // =========================================================================
    predicted_digit_ram u_predicted_digit_ram (
        .clk(clk),
        .wr_en(digit_ram_wr_en),
        .wr_data(predicted_digit),
        .rd_addr(1'b0),  // Always read from address 0
        .rd_data(digit_ram_rd_data)
    );
    
    // =========================================================================
    // UART RX for Digit Reader (separate instance)
    // =========================================================================
    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_digit_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(digit_rx_data),
        .ready(digit_rx_ready)
    );
    
    // =========================================================================
    // UART TX for responses
    // =========================================================================
    uart_tx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_digit_tx (
        .clk(clk),
        .rst(rst),
        .data(tx_data),
        .send(tx_send),
        .tx(tx_out),
        .busy(tx_busy)
    );
    
    // =========================================================================
    // Digit Reader
    // =========================================================================
    digit_reader u_digit_reader (
        .clk(clk),
        .rst(rst),
        .rx_data(digit_rx_data),
        .rx_ready(digit_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(tx_data),
        .tx_send(tx_send),
        .tx_busy(tx_busy)
    );
    
    // =========================================================================
    // Digit Display Reader - Reads from predicted_digit_ram
    // =========================================================================
    digit_display_reader u_digit_display_reader (
        .clk(clk),
        .rst(rst),
        .digit_ram_data(digit_ram_rd_data),
        .display_digit(digit_left)
    );
    
    // =========================================================================
    // LED Control
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            led_reg <= 0;
        end else begin
            // Show loader status when loading, inference result when done
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
        end
    end
    
    assign led = led_reg;
    
    // =========================================================================
    // 7-Segment Display
    // =========================================================================
    // Rightmost digit shows current predicted_digit, leftmost shows stored value from memory
    assign digit_right = predicted_digit;
    
    seven_segment_display u_display (
        .clk(clk),
        .rst(rst),
        .digit_left(digit_left),
        .digit_right(digit_right),
        .seg(seg),
        .an(an)
    );
    
    // Connect TX output
    assign tx = tx_out;

endmodule


