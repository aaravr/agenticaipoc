package com.raghuet.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.modelcontextprotocol.server.transport.WebMvcSseServerTransport;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

public class CleaningSseTransport extends WebMvcSseServerTransport {
    public CleaningSseTransport(ObjectMapper objectMapper, String messageEndpoint) {
        super(objectMapper, messageEndpoint);
    }

}