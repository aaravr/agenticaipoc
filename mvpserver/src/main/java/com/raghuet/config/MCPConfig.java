package com.raghuet.config;

import com.raghuet.service.ClientService;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Service;

@Configuration
public class MCPConfig {

    private final ClientService clientService;

    @Autowired
    public MCPConfig(ClientService clientService) {
        this.clientService = clientService;
    }

    @Bean
    ToolCallbackProvider userTools() {
        return MethodToolCallbackProvider
                .builder()
                .toolObjects(clientService)
                .build();
    }

}
