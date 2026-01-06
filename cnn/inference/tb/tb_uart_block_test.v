`timescale 1ns / 1ps

module tb_debug_logic;

    // Inputs to DUT
    reg clk;
    reg rst;
    reg rx;

    // Instantiate the Top Module
    top dut (
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .tx(),
        .led(),
        .seg(),
        .an()
    );

    // Clock Generation (100MHz)
    always #5 clk = ~clk;

    // =========================================================
    // HELPER TASK: Inject Byte directly into UART Router
    // Bypasses the slow UART RX module to speed up simulation
    // =========================================================
    task inject_byte;
        input [7:0] data_val;
        begin
            // Force the signals inside the Router module
            // Note: These paths must match your hierarchy
            force dut.u_uart_router.rx_data = data_val;
            force dut.u_uart_router.rx_ready = 1;
            #10; // Wait one clock cycle
            force dut.u_uart_router.rx_ready = 0;
            #10; // Wait a bit
            // Release is not strictly necessary for pulses, but good practice
            release dut.u_uart_router.rx_data;
            release dut.u_uart_router.rx_ready;
        end
    endtask

    // =========================================================
    // MAIN TEST SEQUENCE
    // =========================================================
    integer i;
    
    // NEW: Sticky Success Flag
    reg success_detected;
    always @(posedge dut.img_loaded) begin
        success_detected = 1;
        $display("[%t] TESTBENCH: Caught 'img_loaded' pulse!", $time);
    end
    
    initial begin
        // Initialize
        success_detected = 0;
        clk = 0;
        rst = 1;
        rx = 1;
        #100;
        rst = 0;
        #100;

        $display("\n============================================================");
        $display(" STARTING DEBUG SIMULATION");
        $display("============================================================");

        // 2. CHEAT: Force Weights Loaded
        // We skip the weight loading phase to test the Image Logic immediately
        $display("[%t] Forcing 'weights_loaded' to TRUE to skip weight loading...", $time);
        // Force the specific input port of the image_loader
        force dut.u_image_loader.weights_loaded = 1; 
        // Also keep the router force just in case
        force dut.u_uart_router.weights_loaded = 1;
        #100;

        // 3. Send Image 1 Start Marker
        $display("[%t] Sending Image Start Markers (0xBB 0x66)...", $time);
        inject_byte(8'hBB);
        inject_byte(8'h66);

        // 4. Send 784 Pixel Bytes
        $display("[%t] sending 784 dummy pixel bytes...", $time);
        for (i = 0; i < 784; i = i + 1) begin
            inject_byte(8'hCA); // Dummy pixel data
        end

        // 5. Send Image End Markers (THE CRITICAL PART)
        $display("[%t] Sending Image End Markers (0x66 0xBB)...", $time);
        inject_byte(8'h66); // IMG_END1
        inject_byte(8'hBB); // IMG_END2

        // 6. Wait and Check
        #200;
        // CHECK RESULT
        if (success_detected) begin
             $display("\n========================================================");
             $display(" SUCCESS: Image Loaded signal was detected!");
             $display("========================================================\n");
        end else begin
             $display("\n========================================================");
             $display(" FAILURE: Image Loaded signal NEVER went high!");
             $display("========================================================\n");
             $display("DEBUG INFO:");
             $display("Router Byte Count: %d", dut.u_uart_router.byte_count);
             $display("Loader Byte Count: %d", dut.u_image_loader.byte_count);
             $display("Loader State:      %d (0=Receiving, 1=Done)", dut.u_image_loader.state);
        end
        
        $display("============================================================\n");
        $stop;
    end

endmodule