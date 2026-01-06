module top (
    input wire clk,               // 100 MHz System Clock
    input wire rst,               // Reset Button (active high)
    input wire rx,                // UART RX Line
    output wire tx,               // UART TX Line
    output wire [15:0] led,       // Status LEDs
    output wire [6:0] seg,        // 7-segment display segments
    output wire [3:0] an,         // 7-segment display anodes
    input wire [15:0] sw,         // Switches (optional)
    input wire btnU,              // Buttons (optional)
    input wire btnD,
    input wire btnL,
    input wire btnR
);

    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // UART Router signals
    wire [7:0] weight_rx_data;
    wire weight_rx_ready;
    wire [7:0] image_rx_data;
    wire image_rx_ready;
    wire [7:0] cmd_rx_data;
    wire cmd_rx_ready;
    
    // Weight Loader Signals
    wire [12:0] conv_w_wr_addr;
    wire [7:0]  conv_w_wr_data;
    wire        conv_w_wr_en;
    
    wire [5:0]  conv_b_wr_addr;
    wire [31:0] conv_b_wr_data;
    wire        conv_b_wr_en;
    
    wire [12:0] dense_w_wr_addr;
    wire [7:0]  dense_w_wr_data;
    wire        dense_w_wr_en;
    
    wire [3:0]  dense_b_wr_addr;
    wire [31:0] dense_b_wr_data;
    wire        dense_b_wr_en;
    
    wire weights_loaded;
    wire [15:0] loader_led; // DEBUG LEDs
    
    // Inference Signals (Read Side)
    wire [12:0] conv_w_rd_addr;
    wire [7:0]  conv_w_rd_data;
    
    wire [5:0]  conv_b_rd_addr;
    wire [31:0] conv_b_rd_data;
    
    wire [12:0] dense_w_rd_addr;
    wire [7:0]  dense_w_rd_data;
    
    wire [3:0]  dense_b_rd_addr;
    wire [31:0] dense_b_rd_data;

    // Intermediate Buffers
    wire [13:0] buf_a_addr;
    wire [7:0]  buf_a_wr_data;
    wire        buf_a_wr_en;
    wire [7:0]  buf_a_rd_data;

    wire [11:0] buf_b_addr;
    wire [7:0]  buf_b_wr_data;
    wire        buf_b_wr_en;
    wire [7:0]  buf_b_rd_data;
    
    // Image RAM Signals
    wire [9:0]  img_wr_addr;
    wire [7:0]  img_wr_data;
    wire        img_wr_en;
    wire        img_loaded;
    
    wire [9:0]  img_rd_addr;
    wire [7:0]  img_rd_data;
    
    // Control & Results
    wire        start_inference_pulse;
    wire        inference_done;
    wire [3:0]  predicted_digit;
    
    // Class Scores
    wire signed [31:0] class_score_0, class_score_1, class_score_2, class_score_3, class_score_4;
    wire signed [31:0] class_score_5, class_score_6, class_score_7, class_score_8, class_score_9;

    // Edge Detectors
    reg img_loaded_prev;
    reg inference_done_prev;
    
    // RAM Write Enables
    wire digit_ram_wr_en;
    wire scores_ram_wr_en;
    
    // UART TX & Readers
    wire [7:0] digit_ram_rd_data;
    wire [5:0] scores_ram_rd_addr;
    wire [7:0] scores_ram_rd_data;
    
    wire [7:0] tx_data;
    wire       tx_send;
    wire       tx_busy;
    wire       tx_out;
    
    wire [7:0] digit_tx_data;
    wire       digit_tx_send;
    wire [7:0] scores_tx_data;
    wire       scores_tx_send;
    
    // Display
    wire [3:0] digit_left;
    reg [15:0] led_reg;
    reg [26:0] heartbeat_cnt;

    // =========================================================================
    // 1. UART ROUTER
    // =========================================================================
    uart_router u_uart_router (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .weights_loaded(weights_loaded),
        .weight_rx_data(weight_rx_data),
        .weight_rx_ready(weight_rx_ready),
        .image_rx_data(image_rx_data),
        .image_rx_ready(image_rx_ready),
        .cmd_rx_data(cmd_rx_data),
        .cmd_rx_ready(cmd_rx_ready)
    );

    // =========================================================================
    // 2. WEIGHT LOADER
    // =========================================================================
    weight_loader u_weight_loader (
        .clk(clk),
        .rst(rst),
        .rx_data(weight_rx_data),
        .rx_ready(weight_rx_ready),
        .transfer_done(weights_loaded),
        
        // Conv (L1 + L2 combined)
        .conv_w_addr(conv_w_wr_addr),
        .conv_w_data(conv_w_wr_data),
        .conv_w_en(conv_w_wr_en),
        .conv_b_addr(conv_b_wr_addr),
        .conv_b_data(conv_b_wr_data),
        .conv_b_en(conv_b_wr_en),
        
        // Dense
        .dense_w_addr(dense_w_wr_addr),
        .dense_w_data(dense_w_wr_data),
        .dense_w_en(dense_w_wr_en),
        .dense_b_addr(dense_b_wr_addr),
        .dense_b_data(dense_b_wr_data),
        .dense_b_en(dense_b_wr_en),
        
        // Debug
        .led(loader_led)
    );

    // =========================================================================
    // 3. CONV MEMORIES (Weights & Biases)
    // =========================================================================
    conv_weights_ram u_conv_w_ram (
        .clk(clk),
        .wr_addr(conv_w_wr_addr),
        .wr_data(conv_w_wr_data),
        .wr_en(conv_w_wr_en),
        .rd_addr(conv_w_rd_addr),
        .rd_data(conv_w_rd_data)
    );

    conv_biases_ram u_conv_b_ram (
        .clk(clk),
        .wr_addr(conv_b_wr_addr),
        .wr_data(conv_b_wr_data),
        .wr_en(conv_b_wr_en),
        .rd_addr(conv_b_rd_addr),
        .rd_data(conv_b_rd_data)
    );

    // =========================================================================
    // 4. DENSE BIASES
    // =========================================================================
    dense_biases_ram u_dense_b_ram (
        .clk(clk),
        .wr_addr(dense_b_wr_addr),
        .wr_data(dense_b_wr_data),
        .wr_en(dense_b_wr_en),
        .rd_addr(dense_b_rd_addr),
        .rd_data(dense_b_rd_data)
    );

    // =========================================================================
    // 5. CNN RAM (Buffers A/B + Dense Weights)
    // =========================================================================
    wire [12:0] ram_dense_addr = dense_w_wr_en ? dense_w_wr_addr : dense_w_rd_addr;
    wire [7:0]  ram_dense_w_d  = dense_w_wr_data; 
    wire        ram_dense_we   = dense_w_wr_en;

    ram_cnn u_ram_cnn (
        .clk(clk),
        
        // Buffer A
        .buf_a_addr(buf_a_addr),
        .buf_a_wr_data(buf_a_wr_data),
        .buf_a_wr_en(buf_a_wr_en),
        .buf_a_rd_data(buf_a_rd_data),
        
        // Buffer B
        .buf_b_addr(buf_b_addr),
        .buf_b_wr_data(buf_b_wr_data),
        .buf_b_wr_en(buf_b_wr_en),
        .buf_b_rd_data(buf_b_rd_data),
        
        // Dense Weights
        .dw_addr(ram_dense_addr),
        .dw_wr_data(ram_dense_w_d),
        .dw_wr_en(ram_dense_we),
        .dw_rd_data(dense_w_rd_data)
    );

    // =========================================================================
    // 6. IMAGE LOADING & STORAGE
    // =========================================================================
    image_loader u_image_loader (
        .clk(clk),
        .rst(rst),
        .weights_loaded(weights_loaded),
        .rx_data(image_rx_data),
        .rx_ready(image_rx_ready),
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
        .rd_addr(img_rd_addr),
        .rd_data(img_rd_data)
    );

    // =========================================================================
    // 7. INFERENCE ENGINE
    // =========================================================================
    always @(posedge clk) begin
        if (rst) img_loaded_prev <= 0;
        else img_loaded_prev <= img_loaded;
    end
    assign start_inference_pulse = img_loaded && !img_loaded_prev;

    inference u_inference (
        .clk(clk),
        .rst(rst),
        .start(start_inference_pulse),
        .done(inference_done),
        .predicted_digit(predicted_digit),
        
        // Image Read
        .img_addr(img_rd_addr),
        .img_data(img_rd_data),
        
        // Conv Read
        .conv_w_addr(conv_w_rd_addr),
        .conv_w_data(conv_w_rd_data),
        .conv_b_addr(conv_b_rd_addr),
        .conv_b_data(conv_b_rd_data),
        
        // Dense Read
        .dense_w_addr(dense_w_rd_addr),
        .dense_w_data(dense_w_rd_data),
        .dense_b_addr(dense_b_rd_addr),
        .dense_b_data(dense_b_rd_data),
        
        // Buffers
        .buf_a_addr(buf_a_addr),
        .buf_a_wr_data(buf_a_wr_data),
        .buf_a_wr_en(buf_a_wr_en),
        .buf_a_rd_data(buf_a_rd_data),
        
        .buf_b_addr(buf_b_addr),
        .buf_b_wr_data(buf_b_wr_data),
        .buf_b_wr_en(buf_b_wr_en),
        .buf_b_rd_data(buf_b_rd_data),
        
        // Scores
        .class_score_0(class_score_0), .class_score_1(class_score_1),
        .class_score_2(class_score_2), .class_score_3(class_score_3),
        .class_score_4(class_score_4), .class_score_5(class_score_5),
        .class_score_6(class_score_6), .class_score_7(class_score_7),
        .class_score_8(class_score_8), .class_score_9(class_score_9)
    );

    // =========================================================================
    // 8. RESULT STORAGE & READBACK
    // =========================================================================
    always @(posedge clk) begin
        if (rst) inference_done_prev <= 0;
        else inference_done_prev <= inference_done;
    end
    assign digit_ram_wr_en  = inference_done && !inference_done_prev;
    assign scores_ram_wr_en = inference_done && !inference_done_prev;

    predicted_digit_ram u_pred_ram (
        .clk(clk),
        .wr_en(digit_ram_wr_en),
        .wr_data(predicted_digit),
        .rd_addr(1'b0),
        .rd_data(digit_ram_rd_data)
    );

    scores_ram u_scores_ram (
        .clk(clk),
        .wr_en(scores_ram_wr_en),
        .score_0(class_score_0), .score_1(class_score_1),
        .score_2(class_score_2), .score_3(class_score_3),
        .score_4(class_score_4), .score_5(class_score_5),
        .score_6(class_score_6), .score_7(class_score_7),
        .score_8(class_score_8), .score_9(class_score_9),
        .rd_addr(scores_ram_rd_addr),
        .rd_data(scores_ram_rd_data)
    );

    // =========================================================================
    // 9. UART TX & DISPLAY
    // =========================================================================
    uart_tx #(.CLK_FREQ(100_000_000), .BAUD_RATE(115200)) u_uart_tx (
        .clk(clk), .rst(rst),
        .data(tx_data), .send(tx_send), .tx(tx_out), .busy(tx_busy)
    );
    assign tx = tx_out;

    digit_reader u_digit_reader (
        .clk(clk), .rst(rst),
        .rx_data(cmd_rx_data), .rx_ready(cmd_rx_ready),
        .digit_data(digit_ram_rd_data),
        .tx_data(digit_tx_data), .tx_send(digit_tx_send), .tx_busy(tx_busy)
    );

    scores_reader u_scores_reader (
        .clk(clk), .rst(rst),
        .rx_data(cmd_rx_data), .rx_ready(cmd_rx_ready),
        .scores_data(scores_ram_rd_data), .scores_addr(scores_ram_rd_addr),
        .tx_data(scores_tx_data), .tx_send(scores_tx_send), .tx_busy(tx_busy)
    );

    assign tx_data = digit_tx_send ? digit_tx_data : scores_tx_data;
    assign tx_send = digit_tx_send | scores_tx_send;

    digit_display_reader u_disp_reader (
        .clk(clk), .rst(rst),
        .digit_ram_data(digit_ram_rd_data),
        .display_digit(digit_left)
    );

    seven_segment_display u_display (
        .clk(clk), .rst(rst),
        .digit_left(digit_left),
        .digit_right(predicted_digit),
        .seg(seg), .an(an)
    );

    // =========================================================================
    // 10. LED LOGIC
    // =========================================================================
    always @(posedge clk) heartbeat_cnt <= heartbeat_cnt + 1;

    always @(posedge clk) begin
        if (rst) begin
            led_reg <= 0;
        end else begin
            if (!weights_loaded) begin
                // SHOW LOADING PROGRESS
                led_reg <= loader_led;
            end else begin
                // SHOW INFERENCE STATUS
                led_reg[3:0] <= predicted_digit;
                led_reg[4]   <= !inference_done;
                led_reg[5]   <= inference_done;
                led_reg[6]   <= img_loaded;
                led_reg[7]   <= weights_loaded;
                led_reg[15]  <= heartbeat_cnt[26]; 
                led_reg[14:8]<= 0;
            end
        end
    end
    
    assign led = led_reg;

    // =================================================================
    // SIMULATION DEBUG SPIES (Add to bottom of top.v)
    // =================================================================
    // synthesis translate_off

    // 1. Monitor the connection wire
    always @(posedge image_rx_ready) begin
        $display("[%t] [WIRE SPY] image_rx_ready went HIGH! Data: %h", $time, image_rx_data);
    end

    // 2. Monitor the Loader's internal acceptance
    always @(posedge u_image_loader.rx_ready) begin
        $display("[%t] [LOADER SPY] Input rx_ready went HIGH. Internal WeightsLoaded: %b", $time, u_image_loader.weights_loaded);
    end

    // 3. Monitor the Router's internal send
    always @(posedge u_uart_router.image_rx_ready) begin
        $display("[%t] [ROUTER SPY] Output image_rx_ready went HIGH.", $time);
    end

    // synthesis translate_on

endmodule