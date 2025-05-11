package com.raghuet.config;

import com.raghuet.service.ClientService;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Configuration
public class MCPConfig {
    private static final Logger logger = LoggerFactory.getLogger(MCPConfig.class);

    private final ClientService clientService;

    @Autowired
    public MCPConfig(ClientService clientService) {
        this.clientService = clientService;
    }

    @Bean
    @Primary
    ToolCallbackProvider userTools() {
        return MethodToolCallbackProvider
                .builder()
                .toolObjects(clientService)
                .build();
    }
}
