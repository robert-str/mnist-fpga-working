module weight_loader (
    input wire clk,
    input wire rst,
    input wire [7:0] rx_data,
    input wire rx_ready,
    output reg transfer_done,
    output reg [15:0] led,
    
    // Conv Weights (Shared L1+L2)
    output reg [12:0] conv_w_addr,
    output reg [7:0] conv_w_data,
    output reg conv_w_en,
    
    // Conv Biases (Shared L1+L2)
    output reg [5:0] conv_b_addr,
    output reg [31:0] conv_b_data,
    output reg conv_b_en,

    // Dense Weights
    output reg [12:0] dense_w_addr,
    output reg [7:0] dense_w_data,
    output reg dense_w_en,

    // Dense Biases
    output reg [3:0] dense_b_addr,
    output reg [31:0] dense_b_data,
    output reg dense_b_en
);
    // Address Map
    localparam L1_W_END = 144;
    localparam L1_B_END = 144 + 64;
    localparam L2_W_END = L1_B_END + 4608;
    localparam L2_B_END = L2_W_END + 128;
    localparam DN_W_END = L2_B_END + 8000;
    localparam TOTAL    = DN_W_END + 40;

    reg [15:0] global_addr;
    reg [1:0] byte_idx;
    reg [31:0] bias_accum;

    always @(posedge clk) begin
        if (rst) begin
            transfer_done <= 0;
            global_addr <= 0;
            byte_idx <= 0;
            conv_w_en <= 0; conv_b_en <= 0; dense_w_en <= 0; dense_b_en <= 0;
            led <= 0;
        end else if (rx_ready) begin
            // 1. L1 Weights (0-143)
            if (global_addr < L1_W_END) begin
                conv_w_addr <= global_addr;
                conv_w_data <= rx_data;
                conv_w_en <= 1;
            end 
            // 2. L1 Biases (144-207)
            else if (global_addr < L1_B_END) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    conv_b_addr <= (global_addr - L1_W_END) >> 2;
                    conv_b_data <= {rx_data, bias_accum[23:0]};
                    conv_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end
            // 3. L2 Weights (208-4815)
            else if (global_addr < L2_W_END) begin
                conv_w_addr <= 144 + (global_addr - L1_B_END);
                conv_w_data <= rx_data;
                conv_w_en <= 1;
            end
            // 4. L2 Biases (4816-4943)
            else if (global_addr < L2_B_END) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    conv_b_addr <= 16 + ((global_addr - L2_W_END) >> 2);
                    conv_b_data <= {rx_data, bias_accum[23:0]};
                    conv_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end
            // 5. Dense Weights (4944-12943)
            else if (global_addr < DN_W_END) begin
                dense_w_addr <= (global_addr - L2_B_END);
                dense_w_data <= rx_data;
                dense_w_en <= 1;
            end
            // 6. Dense Biases (12944-12983)
            else if (global_addr < TOTAL) begin
                bias_accum[byte_idx*8 +: 8] <= rx_data;
                if (byte_idx == 3) begin
                    dense_b_addr <= (global_addr - DN_W_END) >> 2;
                    dense_b_data <= {rx_data, bias_accum[23:0]};
                    dense_b_en <= 1;
                    byte_idx <= 0;
                end else byte_idx <= byte_idx + 1;
            end

            // Update LED progress
            led <= global_addr[15:0];
            
            global_addr <= global_addr + 1;
            if (global_addr >= TOTAL - 1) begin
                transfer_done <= 1;
                led <= 16'hFFFF;
            end
        end else begin
            conv_w_en <= 0; conv_b_en <= 0; dense_w_en <= 0; dense_b_en <= 0;
        end
    end
endmodule
