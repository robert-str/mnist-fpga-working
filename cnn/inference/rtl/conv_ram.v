// Combined Conv Weights RAM (L1 + L2)
// L1: 0 - 143 (144 bytes)
// L2: 144 - 4751 (4608 bytes)
// Total: 4752 bytes (requires 13-bit address: 2^13 = 8192 > 4752)
module conv_weights_ram (
    input wire clk,
    input wire [12:0] wr_addr,  // FIXED: Was [12:0], correct
    input wire [7:0] wr_data,
    input wire wr_en,
    input wire [12:0] rd_addr,  // FIXED: Was [12:0], correct
    output reg [7:0] rd_data
);
    (* ram_style = "block" *) reg [7:0] ram [0:8191];  // FIXED: Expanded to full 2^13
    
    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end
    
    // Synchronous read
    always @(posedge clk) begin
        rd_data <= ram[rd_addr];
    end
endmodule

// Combined Conv Biases RAM (L1 + L2)
// L1: 0 - 15
// L2: 16 - 47
// Total: 48 values (192 bytes)
module conv_biases_ram (
    input wire clk,
    input wire [5:0] wr_addr,
    input wire [31:0] wr_data,
    input wire wr_en,
    input wire [5:0] rd_addr,
    output reg [31:0] rd_data
);
    (* ram_style = "distributed" *) reg [31:0] ram [0:47];
    always @(posedge clk) begin
        if (wr_en) ram[wr_addr] <= wr_data;
        rd_data <= ram[rd_addr];
    end
endmodule
