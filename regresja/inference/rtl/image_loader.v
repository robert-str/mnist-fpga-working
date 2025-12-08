/*
================================================================================
Image Loader - Receives 784-byte image via UART
================================================================================
Protocol: 0xBB 0x66, 784 bytes, 0x66 0xBB
================================================================================
*/

module image_loader (
    input wire clk,
    input wire rst,
    input wire rx,
    input wire weights_loaded,    // Only accept images after weights loaded
    
    output reg [9:0] wr_addr,
    output reg [7:0] wr_data,
    output reg wr_en,
    output reg image_loaded
);

    // Protocol: 0xBB 0x66, 784 bytes, 0x66 0xBB
    localparam IMG_START1 = 8'hBB;
    localparam IMG_START2 = 8'h66;
    localparam IMG_END1 = 8'h66;
    localparam IMG_END2 = 8'hBB;
    localparam IMG_SIZE = 784;

    // States
    localparam STATE_WAIT_START1 = 3'd0;
    localparam STATE_WAIT_START2 = 3'd1;
    localparam STATE_RECEIVING   = 3'd2;
    localparam STATE_DONE        = 3'd3;

    // UART receiver
    wire [7:0] rx_data;
    wire rx_ready;
    
    // Instantiate separate UART receiver for image loading
    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_img_rx (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .data(rx_data),
        .ready(rx_ready)
    );
    
    reg [2:0] state;
    reg [9:0] byte_count;
    reg [7:0] prev_byte;

    always @(posedge clk) begin
        if (rst) begin
            state <= STATE_WAIT_START1;
            wr_addr <= 0;
            wr_data <= 0;
            wr_en <= 0;
            byte_count <= 0;
            prev_byte <= 0;
            image_loaded <= 0;
        end else begin
            wr_en <= 0;
            image_loaded <= 0;
            
            // Only process if weights are loaded
            if (weights_loaded) begin
                case (state)
                    STATE_WAIT_START1: begin
                        if (rx_ready && rx_data == IMG_START1) begin
                            state <= STATE_WAIT_START2;
                        end
                    end
                    
                    STATE_WAIT_START2: begin
                        if (rx_ready) begin
                            if (rx_data == IMG_START2) begin
                                state <= STATE_RECEIVING;
                                byte_count <= 0;
                                prev_byte <= 0;
                            end else if (rx_data == IMG_START1) begin
                                state <= STATE_WAIT_START2;
                            end else begin
                                state <= STATE_WAIT_START1;
                            end
                        end
                    end
                    
                    STATE_RECEIVING: begin
                        if (rx_ready) begin
                            // First, store the previous byte if we have data pending
                            if (byte_count > 0 && byte_count <= IMG_SIZE) begin
                                wr_addr <= byte_count - 1;
                                wr_data <= prev_byte;
                                wr_en <= 1;
                            end
                            
                            // Check for end marker after storing
                            if (prev_byte == IMG_END1 && rx_data == IMG_END2 && byte_count >= IMG_SIZE) begin
                                state <= STATE_DONE;
                                image_loaded <= 1;
                            end else begin
                                // Advance counter and store current byte for next iteration
                                byte_count <= byte_count + 1;
                                prev_byte <= rx_data;
                            end
                        end
                    end
                    
                    STATE_DONE: begin
                        // Go back to waiting for next image
                        state <= STATE_WAIT_START1;
                    end
                endcase
            end
        end
    end

endmodule


