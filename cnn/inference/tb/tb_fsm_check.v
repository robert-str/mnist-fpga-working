`timescale 1ns / 1ps

module tb_fsm_retrigger;

    // --- Inputs to DUT ---
    reg clk;
    reg rst;
    reg start;

    // --- Outputs from DUT ---
    wire done;
    wire [3:0] predicted_digit;

    // --- Internal RAM & Signals ---
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

    // Dummy RAMs
    reg [7:0] image_ram [0:783];
    reg [7:0] conv_weights_ram [0:8191];
    reg [31:0] conv_biases_ram [0:47];
    reg [7:0] dense_weights_ram [0:7999];
    reg [31:0] dense_biases_ram [0:9];
    reg [7:0] buf_a_ram [0:10815];
    reg [7:0] buf_b_ram [0:2703];

    // RAM Assignments
    assign img_data = image_ram[img_addr];
    assign conv_w_data = conv_weights_ram[conv_w_addr];
    assign conv_b_data = conv_biases_ram[conv_b_addr];
    assign dense_w_data = dense_weights_ram[dense_w_addr];
    assign dense_b_data = dense_biases_ram[dense_b_addr];
    assign buf_a_rd_data = buf_a_ram[buf_a_addr];
    assign buf_b_rd_data = buf_b_ram[buf_b_addr];

    // Buffer Writes
    always @(posedge clk) begin
        if (buf_a_wr_en) buf_a_ram[buf_a_addr] <= buf_a_wr_data;
        if (buf_b_wr_en) buf_b_ram[buf_b_addr] <= buf_b_wr_data;
    end

    // --- DUT Instantiation ---
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
        .class_score_0(), .class_score_1(), .class_score_2(), .class_score_3(),
        .class_score_4(), .class_score_5(), .class_score_6(), .class_score_7(),
        .class_score_8(), .class_score_9()
    );

    // --- Clock Generation ---
    always #5 clk = ~clk;

    // --- Timeout Logic Helpers ---
    integer timeout_counter;
    localparam MAX_TIMEOUT = 1000000; // Cycles to wait

    // --- Main Test Sequence ---
    initial begin
        $display("============================================================");
        $display(" TEST: FSM Retriggering Check (Stuck DONE State)");
        $display("============================================================");
        
        // Initialize
        clk = 0; rst = 1; start = 0;
        #50 rst = 0;
        #50;

        // -------------------------------------------------------------
        // RUN 1: Start First Inference
        // -------------------------------------------------------------
        $display("[%t] Starting Run 1...", $time);
        @(posedge clk); start = 1;
        @(posedge clk); start = 0;

        // Wait for Done (Simple blocking wait)
        wait(done);
        $display("[%t] Run 1 Finished! (DONE went High)", $time);

        // -------------------------------------------------------------
        // CHECK 1: Does DONE go Low?
        // -------------------------------------------------------------
        // Wait 5 cycles. The FSM should transition DONE -> IDLE, making done=0.
        repeat(5) @(posedge clk);
        
        if (done === 1) begin
             $display("\n[FAILURE] 'done' signal stuck HIGH after Run 1!");
             $display("Reason: FSM is likely trapped in the DONE state.");
             $stop;
        end else begin
             $display("[%t] 'done' signal successfully cleared.", $time);
        end

        // -------------------------------------------------------------
        // RUN 2: Attempt Second Inference
        // -------------------------------------------------------------
        $display("[%t] Starting Run 2...", $time);
        @(posedge clk); start = 1;
        @(posedge clk); start = 0;

        // -------------------------------------------------------------
        // WAIT LOOP WITH TIMEOUT (Replaces fork/join_any)
        // -------------------------------------------------------------
        timeout_counter = 0;
        
        // Wait while DONE is low AND we haven't timed out
        while (done === 0 && timeout_counter < MAX_TIMEOUT) begin
            @(posedge clk);
            timeout_counter = timeout_counter + 1;
        end

        if (timeout_counter >= MAX_TIMEOUT) begin
            $display("\n[FAILURE] Run 2 Timed out! FSM did not restart.");
            $stop;
        end else begin
            $display("[%t] Run 2 Finished! (DONE went High)", $time);
            $display("\n[SUCCESS] FSM successfully re-triggered!");
        end
        
        $finish;
    end

endmodule