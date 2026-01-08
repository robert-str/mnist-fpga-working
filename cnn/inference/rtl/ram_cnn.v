module ram_cnn (
    input clk,
    // Buffer A: Holds L1 Output (16*26*26 = 10,816 bytes)
    input [13:0] buf_a_addr,
    input [7:0] buf_a_wr_data,
    input buf_a_wr_en,
    output wire [7:0] buf_a_rd_data, // Changed to wire for combinational

    // Buffer B: Holds L1 Pool Out (16*13*13 = 2704) OR L2 Pool Out (32*5*5 = 800)
    input [11:0] buf_b_addr,
    input [7:0] buf_b_wr_data,
    input buf_b_wr_en,
    output wire [7:0] buf_b_rd_data, // Changed to wire for combinational

    // Dense Weights: 800 * 10 = 8000 bytes
    // Note: Dense weights can remain synchronous if the FSM handles it, 
    // but let's make them distributed too for consistency with TB.
    input [12:0] dw_addr,
    input [7:0] dw_wr_data,
    input dw_wr_en,
    output wire [7:0] dw_rd_data     // Changed to wire for combinational
);
    // Force Distributed RAM (LUTRAM) for 0-cycle read latency
    (* ram_style = "distributed" *) reg [7:0] buffer_a [0:10815];
    (* ram_style = "distributed" *) reg [7:0] buffer_b [0:2703];
    (* ram_style = "distributed" *) reg [7:0] dense_ram [0:7999];

    // Synchronous Write
    always @(posedge clk) begin
        if (buf_a_wr_en) buffer_a[buf_a_addr] <= buf_a_wr_data;
    end
    
    // Asynchronous (Combinational) Read - Matches Testbench
    assign buf_a_rd_data = buffer_a[buf_a_addr];

    // Synchronous Write
    always @(posedge clk) begin
        if (buf_b_wr_en) buffer_b[buf_b_addr] <= buf_b_wr_data;
    end
    
    // Asynchronous Read
    assign buf_b_rd_data = buffer_b[buf_b_addr];

    // Synchronous Write
    always @(posedge clk) begin
        if (dw_wr_en) dense_ram[dw_addr] <= dw_wr_data;
    end
    
    // Asynchronous Read
    assign dw_rd_data = dense_ram[dw_addr];

endmodule
