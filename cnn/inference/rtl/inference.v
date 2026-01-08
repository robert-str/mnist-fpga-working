/*
================================================================================
Inference Module - Fixed for COMBINATIONAL BRAM reads
================================================================================
Version 3.0 - Fixes L1/L2 Pool timing for immediate RAM reads

Key fix: Pool logic now reads data in the SAME cycle the address is set
(because combinational reads are immediate)
================================================================================
*/

module inference (
    input wire clk,
    input wire rst,
    input wire start,
    output reg done,
    output reg [3:0] predicted_digit,

    // Image RAM
    output reg [9:0] img_addr,
    input wire [7:0] img_data,

    // Conv RAMs (Shared)
    output reg [12:0] conv_w_addr,
    input wire [7:0] conv_w_data,
    output reg [5:0] conv_b_addr,
    input wire [31:0] conv_b_data,

    // Dense RAMs
    output reg [12:0] dense_w_addr,
    input wire [7:0] dense_w_data,
    output reg [3:0] dense_b_addr,
    input wire [31:0] dense_b_data,

    // Buffer A
    output reg [13:0] buf_a_addr,
    output reg [7:0] buf_a_wr_data,
    output reg buf_a_wr_en,
    input wire [7:0] buf_a_rd_data,

    // Buffer B
    output reg [11:0] buf_b_addr,
    output reg [7:0] buf_b_wr_data,
    output reg buf_b_wr_en,
    input wire [7:0] buf_b_rd_data,

    // Scores
    output reg signed [31:0] class_score_0,
    output reg signed [31:0] class_score_1,
    output reg signed [31:0] class_score_2,
    output reg signed [31:0] class_score_3,
    output reg signed [31:0] class_score_4,
    output reg signed [31:0] class_score_5,
    output reg signed [31:0] class_score_6,
    output reg signed [31:0] class_score_7,
    output reg signed [31:0] class_score_8,
    output reg signed [31:0] class_score_9
);

    // States
    localparam IDLE              = 5'd0;
    localparam L1_LOAD_BIAS      = 5'd1;
    localparam L1_LOAD_BIAS_WAIT = 5'd2;
    localparam L1_PREFETCH       = 5'd3;
    localparam L1_CONV           = 5'd4;
    localparam L1_SAVE           = 5'd5;
    localparam L1_POOL           = 5'd6;
    localparam L2_LOAD_BIAS      = 5'd7;
    localparam L2_LOAD_BIAS_WAIT = 5'd8;
    localparam L2_PREFETCH       = 5'd9;
    localparam L2_CONV           = 5'd10;
    localparam L2_SAVE           = 5'd11;
    localparam L2_POOL           = 5'd12;
    localparam DENSE_LOAD_BIAS      = 5'd13;
    localparam DENSE_LOAD_BIAS_WAIT = 5'd14;
    localparam DENSE_PREFETCH       = 5'd15;
    localparam DENSE_MULT           = 5'd16;
    localparam DENSE_NEXT           = 5'd17;
    localparam DONE_STATE           = 5'd18;
    
    reg [4:0] state;

    // Iterators
    reg [5:0] f_idx;
    reg [4:0] ch_idx;
    reg [4:0] r, c;
    reg [1:0] kr, kc;
    reg [3:0] class_idx;
    
    reg signed [31:0] acc;
    reg signed [31:0] temp_val;
    reg [7:0] max_val;
    reg [11:0] flat_idx;
    reg [2:0] pool_step; 
    reg signed [31:0] max_score;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            buf_a_wr_en <= 0; 
            buf_b_wr_en <= 0;
            predicted_digit <= 0;
            img_addr <= 0; conv_w_addr <= 0; conv_b_addr <= 0;
            dense_w_addr <= 0; dense_b_addr <= 0;
            buf_a_addr <= 0; buf_a_wr_data <= 0;
            buf_b_addr <= 0; buf_b_wr_data <= 0;
            class_score_0 <= 0; class_score_1 <= 0; class_score_2 <= 0; class_score_3 <= 0;
            class_score_4 <= 0; class_score_5 <= 0; class_score_6 <= 0; class_score_7 <= 0;
            class_score_8 <= 0; class_score_9 <= 0;
            f_idx <= 0; ch_idx <= 0; r <= 0; c <= 0; kr <= 0; kc <= 0;
            class_idx <= 0;
            acc <= 0; flat_idx <= 0; pool_step <= 0; 
            max_score <= 32'h80000000;
            max_val <= 0;
            temp_val <= 0;
        end else begin
            buf_a_wr_en <= 0; 
            buf_b_wr_en <= 0;
            
            case (state)
            
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= L1_LOAD_BIAS;
                        f_idx <= 0; r <= 0; c <= 0;
                        max_score <= 32'h80000000;
                    end
                end

                // =============================================================
                // LAYER 1 CONV
                // =============================================================
                
                L1_LOAD_BIAS: begin
                    conv_b_addr <= f_idx;            
                    state <= L1_LOAD_BIAS_WAIT;      
                end

                L1_LOAD_BIAS_WAIT: begin
                    acc <= $signed(conv_b_data);     
                    kr <= 0; kc <= 0;
                    img_addr <= (r + 0) * 28 + (c + 0);
                    conv_w_addr <= f_idx * 9 + 0;
                    state <= L1_PREFETCH;
                end

                L1_PREFETCH: begin
                    state <= L1_CONV;
                end

                L1_CONV: begin
                    acc <= acc + $signed({{24{img_data[7]}}, img_data}) * 
                                 $signed({{24{conv_w_data[7]}}, conv_w_data});
                    
                    if (kr == 2 && kc == 2) begin
                        state <= L1_SAVE;
                    end else begin
                        if (kc == 2) begin
                            kc <= 0;
                            kr <= kr + 1;
                            img_addr <= (r + kr + 1) * 28 + (c + 0);
                            conv_w_addr <= f_idx * 9 + (kr + 1) * 3 + 0;
                        end else begin
                            kc <= kc + 1;
                            img_addr <= (r + kr) * 28 + (c + kc + 1);
                            conv_w_addr <= f_idx * 9 + kr * 3 + (kc + 1);
                        end
                    end
                end

                L1_SAVE: begin
                    temp_val = acc >>> 8; 
                    if (temp_val < 0) temp_val = 0;
                    if (temp_val > 127) temp_val = 127;
                    
                    buf_a_addr <= f_idx * 676 + r * 26 + c;
                    buf_a_wr_data <= temp_val[7:0];
                    buf_a_wr_en <= 1;

                    if (c == 25) begin
                        c <= 0;
                        if (r == 25) begin
                            r <= 0;
                            if (f_idx == 15) begin
                                state <= L1_POOL;
                                f_idx <= 0; r <= 0; c <= 0; pool_step <= 0;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= L1_LOAD_BIAS;
                            end
                        end else begin
                            r <= r + 1;
                            state <= L1_LOAD_BIAS;
                        end
                    end else begin
                        c <= c + 1;
                        state <= L1_LOAD_BIAS;
                    end
                end

                // =============================================================
                // L1 POOL - Fixed for COMBINATIONAL reads
                // =============================================================
                // With combinational reads, data is available IMMEDIATELY when
                // address is set. So we read in the same cycle we set the address.
                //
                // Pool steps:
                //   0: Set addr v0, read & store v0 as max
                //   1: Set addr v1, compare with max
                //   2: Set addr v2, compare with max
                //   3: Set addr v3, compare with max, write result
                // =============================================================
                L1_POOL: begin
                    case(pool_step)
                        0: begin 
                            // Set address for v0 and read it immediately
                            buf_a_addr <= f_idx*676 + (r*2)*26 + (c*2);
                            // With combinational read, buf_a_rd_data is now v0
                            // But we need to wait 1 cycle for addr to propagate
                            pool_step <= 1; 
                        end
                        1: begin 
                            // Now buf_a_rd_data has v0
                            max_val <= buf_a_rd_data;
                            // Set address for v1
                            buf_a_addr <= f_idx*676 + (r*2)*26 + (c*2+1);
                            pool_step <= 2; 
                        end
                        2: begin 
                            // buf_a_rd_data has v1
                            if (buf_a_rd_data > max_val) max_val <= buf_a_rd_data;
                            // Set address for v2
                            buf_a_addr <= f_idx*676 + (r*2+1)*26 + (c*2);
                            pool_step <= 3; 
                        end
                        3: begin 
                            // buf_a_rd_data has v2
                            if (buf_a_rd_data > max_val) max_val <= buf_a_rd_data;
                            // Set address for v3
                            buf_a_addr <= f_idx*676 + (r*2+1)*26 + (c*2+1);
                            pool_step <= 4; 
                        end
                        4: begin
                            // buf_a_rd_data has v3
                            // Compare v3 with max and write result
                            buf_b_addr <= f_idx*169 + r*13 + c;
                            buf_b_wr_en <= 1;
                            if (buf_a_rd_data > max_val)
                                buf_b_wr_data <= buf_a_rd_data;
                            else
                                buf_b_wr_data <= max_val;
                            
                            pool_step <= 0;
                            
                            // Advance to next pool position
                            if (c == 12) begin
                                c <= 0;
                                if (r == 12) begin
                                    r <= 0;
                                    if (f_idx == 15) begin
                                        state <= L2_LOAD_BIAS;
                                        f_idx <= 0; r <= 0; c <= 0;
                                    end else begin
                                        f_idx <= f_idx + 1;
                                    end
                                end else begin
                                    r <= r + 1;
                                end
                            end else begin
                                c <= c + 1;
                            end
                        end
                    endcase
                end

                // =============================================================
                // LAYER 2 CONV
                // =============================================================
                
                L2_LOAD_BIAS: begin
                    conv_b_addr <= 16 + f_idx;
                    state <= L2_LOAD_BIAS_WAIT;
                end

                L2_LOAD_BIAS_WAIT: begin
                    acc <= $signed(conv_b_data); 
                    ch_idx <= 0; kr <= 0; kc <= 0;
                    buf_b_addr <= 0 * 169 + (r + 0) * 13 + (c + 0);
                    conv_w_addr <= 144 + f_idx * 144 + 0 * 9 + 0;
                    state <= L2_PREFETCH;
                end
                
                L2_PREFETCH: begin
                    state <= L2_CONV;
                end

                L2_CONV: begin
                    acc <= acc + $signed({{24{buf_b_rd_data[7]}}, buf_b_rd_data}) * 
                                 $signed({{24{conv_w_data[7]}}, conv_w_data});
                    
                    if (kr == 2 && kc == 2 && ch_idx == 15) begin
                        state <= L2_SAVE;
                    end else begin
                        if (kc == 2) begin
                            kc <= 0;
                            if (kr == 2) begin
                                kr <= 0;
                                ch_idx <= ch_idx + 1;
                                buf_b_addr <= (ch_idx + 1) * 169 + (r + 0) * 13 + (c + 0);
                                conv_w_addr <= 144 + f_idx * 144 + (ch_idx + 1) * 9 + 0;
                            end else begin
                                kr <= kr + 1;
                                buf_b_addr <= ch_idx * 169 + (r + kr + 1) * 13 + (c + 0);
                                conv_w_addr <= 144 + f_idx * 144 + ch_idx * 9 + (kr + 1) * 3 + 0;
                            end
                        end else begin
                            kc <= kc + 1;
                            buf_b_addr <= ch_idx * 169 + (r + kr) * 13 + (c + kc + 1);
                            conv_w_addr <= 144 + f_idx * 144 + ch_idx * 9 + kr * 3 + (kc + 1);
                        end
                    end
                end

                L2_SAVE: begin
                    temp_val = acc >>> 8; 
                    if (temp_val < 0) temp_val = 0;
                    if (temp_val > 127) temp_val = 127;
                    
                    buf_a_addr <= f_idx * 121 + r * 11 + c;
                    buf_a_wr_data <= temp_val[7:0];
                    buf_a_wr_en <= 1;

                    if (c == 10) begin
                        c <= 0;
                        if (r == 10) begin
                            r <= 0;
                            if (f_idx == 31) begin
                                state <= L2_POOL;
                                f_idx <= 0; r <= 0; c <= 0; pool_step <= 0;
                            end else begin
                                f_idx <= f_idx + 1;
                                state <= L2_LOAD_BIAS;
                            end
                        end else begin
                            r <= r + 1;
                            state <= L2_LOAD_BIAS;
                        end
                    end else begin
                        c <= c + 1;
                        state <= L2_LOAD_BIAS;
                    end
                end

                // =============================================================
                // L2 POOL - Fixed for COMBINATIONAL reads
                // =============================================================
                L2_POOL: begin
                    case(pool_step)
                        0: begin 
                            buf_a_addr <= f_idx*121 + (r*2)*11 + (c*2);
                            pool_step <= 1; 
                        end
                        1: begin 
                            max_val <= buf_a_rd_data;
                            buf_a_addr <= f_idx*121 + (r*2)*11 + (c*2+1);
                            pool_step <= 2; 
                        end
                        2: begin 
                            if (buf_a_rd_data > max_val) max_val <= buf_a_rd_data;
                            buf_a_addr <= f_idx*121 + (r*2+1)*11 + (c*2);
                            pool_step <= 3; 
                        end
                        3: begin 
                            if (buf_a_rd_data > max_val) max_val <= buf_a_rd_data;
                            buf_a_addr <= f_idx*121 + (r*2+1)*11 + (c*2+1);
                            pool_step <= 4; 
                        end
                        4: begin
                            buf_b_addr <= f_idx*25 + r*5 + c;
                            buf_b_wr_en <= 1;
                            if (buf_a_rd_data > max_val)
                                buf_b_wr_data <= buf_a_rd_data;
                            else
                                buf_b_wr_data <= max_val;
                            
                            pool_step <= 0;
                            
                            if (c == 4) begin
                                c <= 0;
                                if (r == 4) begin
                                    r <= 0;
                                    if (f_idx == 31) begin
                                        state <= DENSE_LOAD_BIAS;
                                        class_idx <= 0;
                                    end else begin
                                        f_idx <= f_idx + 1;
                                    end
                                end else begin
                                    r <= r + 1;
                                end
                            end else begin
                                c <= c + 1;
                            end
                        end
                    endcase
                end

                // =============================================================
                // DENSE LAYER
                // =============================================================
                
                DENSE_LOAD_BIAS: begin
                    dense_b_addr <= class_idx;
                    state <= DENSE_LOAD_BIAS_WAIT;
                end

                DENSE_LOAD_BIAS_WAIT: begin
                    acc <= $signed(dense_b_data); 
                    flat_idx <= 0;
                    buf_b_addr <= 0;
                    dense_w_addr <= class_idx * 800 + 0;
                    state <= DENSE_PREFETCH;
                end

                DENSE_PREFETCH: begin
                    state <= DENSE_MULT;
                end

                DENSE_MULT: begin
                    acc <= acc + $signed({{24{buf_b_rd_data[7]}}, buf_b_rd_data}) * 
                                 $signed({{24{dense_w_data[7]}}, dense_w_data});

                    if (flat_idx == 799) begin
                        state <= DENSE_NEXT;
                    end else begin
                        flat_idx <= flat_idx + 1;
                        buf_b_addr <= flat_idx + 1;
                        dense_w_addr <= class_idx * 800 + (flat_idx + 1);
                    end
                end

                DENSE_NEXT: begin
                    case (class_idx)
                        0: class_score_0 <= acc;
                        1: class_score_1 <= acc;
                        2: class_score_2 <= acc;
                        3: class_score_3 <= acc;
                        4: class_score_4 <= acc;
                        5: class_score_5 <= acc;
                        6: class_score_6 <= acc;
                        7: class_score_7 <= acc;
                        8: class_score_8 <= acc;
                        9: class_score_9 <= acc;
                    endcase

                    if (class_idx == 0 || acc > max_score) begin
                        max_score <= acc;
                        predicted_digit <= class_idx;
                    end

                    if (class_idx == 9) state <= DONE_STATE;
                    else begin
                        class_idx <= class_idx + 1;
                        state <= DENSE_LOAD_BIAS;
                    end
                end

                DONE_STATE: begin
                    done <= 1;       
                    state <= IDLE;   
                end
                
                default: state <= IDLE;
                
            endcase
        end
    end
    
    // Debug
    // synthesis translate_off
    initial begin
        $display("");
        $display("==========================================================");
        $display("[INFERENCE] VERSION 3.0 - Fixed Pool Timing");
        $display("[INFERENCE] Pool now correctly handles combinational reads");
        $display("==========================================================");
        $display("");
    end
    // synthesis translate_on

endmodule