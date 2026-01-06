// Combined Conv Weights RAM (L1 + L2)
// L1: 0 - 143 (144 bytes)
// L2: 144 - 4751 (4608 bytes)
module conv_weights_ram (
    input wire clk,
    input wire [12:0] wr_addr,
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [12:0] rd_addr,
    output wire [7:0] rd_data  // Changed to wire
);
    // Force Distributed RAM for Combinational Read
    (* ram_style = "distributed" *) reg [7:0] ram [0:8191];
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end
    
    // Asynchronous read (Matches TB)
    assign rd_data = ram[rd_addr];
endmodule

// Combined Conv Biases RAM (L1 + L2)
module conv_biases_ram (
    input wire clk,
    input wire [5:0] wr_addr,
    input wire [31:0] wr_data,
    input wire wr_en,
    input wire [5:0] rd_addr,
    output wire [31:0] rd_data // Changed to wire
);
    // Force Distributed RAM
    (* ram_style = "distributed" *) reg [31:0] ram [0:47];

    always @(posedge clk) begin
        if (wr_en) ram[wr_addr] <= wr_data;
    end

    // Asynchronous read
    assign rd_data = ram[rd_addr];
endmodule