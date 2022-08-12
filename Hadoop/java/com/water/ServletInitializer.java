package com.water;

import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.web.support.SpringBootServletInitializer;

import com.water.calibrate.controller.CalibrateSwatServer;

public class ServletInitializer extends SpringBootServletInitializer{
	@Override
    protected SpringApplicationBuilder configure(SpringApplicationBuilder builder) {
        return builder.sources(CalibrateSwatServer.class);
        //此处的CalibrateSwatServer是springboot启动类
    }
}
