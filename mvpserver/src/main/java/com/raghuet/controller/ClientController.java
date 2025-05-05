package com.raghuet.controller;


import com.raghuet.model.ClientInfo;
import com.raghuet.model.ClientStatus;
import com.raghuet.service.ClientService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class ClientController {

    private final ClientService clientService;

    @PostMapping("/capture")
    public ResponseEntity<ClientInfo> captureClientInfo(@RequestBody ClientInfo clientInfo) {
        return ResponseEntity.ok(clientService.saveClientInfo(clientInfo));
    }

    @PostMapping("/outreach")
    public ResponseEntity<ClientInfo> updateClientInfo(@RequestBody ClientInfo clientInfo) {
        return ResponseEntity.ok(clientService.updateClientInfo(clientInfo));
    }

    @PostMapping("/qa-verify")
    public ResponseEntity<ClientInfo> qaVerify(@RequestBody ClientInfo clientInfo) {
        return ResponseEntity.ok(clientService.qaVerify(clientInfo));
    }

    @PostMapping("/approve")
    public ResponseEntity<ClientInfo> approve(@RequestBody ClientInfo clientInfo) {
        return ResponseEntity.ok(clientService.approve(clientInfo));
    }

    @GetMapping("/client/{id}")
    public ResponseEntity<ClientInfo> getClientInfo(@PathVariable String id) {
        return ResponseEntity.ok(clientService.getClientInfo(id));
    }

    @GetMapping("/clients/status/{status}")
    public ResponseEntity<List<ClientInfo>> getClientsByStatus(@PathVariable ClientStatus status) {
        return ResponseEntity.ok(clientService.getClientsByStatus(status));
    }

}