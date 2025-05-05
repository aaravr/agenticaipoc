package com.raghuet.service;

import com.raghuet.ClientRepository;
import com.raghuet.exception.ClientNotFoundException;
import com.raghuet.model.ApprovalStatus;
import com.raghuet.model.ClientInfo;
import com.raghuet.model.ClientStatus;
import com.raghuet.model.QAStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ClientService {

//    private final RestTemplate restTemplate;
//    private final String BASE_URL = "https://dummyjson.com";

    @Autowired
    private ClientRepository clientRepository;

    @Tool(name="saveClientInfo", description = "Save or create new client details")
    @Transactional
    public ClientInfo saveClientInfo(ClientInfo clientInfo) {
        clientInfo.setStatus(ClientStatus.NEW);
        clientInfo.setQaStatus(QAStatus.NOT_STARTED);
        clientInfo.setApprovalStatus(ApprovalStatus.NOT_STARTED);
        return clientRepository.save(clientInfo);
    }

    @Tool(name="updateClientInfo", description = "update client details")
    @Transactional
    public ClientInfo updateClientInfo(ClientInfo updatedInfo) {
        ClientInfo existingInfo = clientRepository.findById(updatedInfo.getId())
                .orElseThrow(() -> new ClientNotFoundException("Client not found"));

        // Update fields
        existingInfo.setName(updatedInfo.getName());
        existingInfo.setEmail(updatedInfo.getEmail());
        existingInfo.setPhone(updatedInfo.getPhone());
        existingInfo.setCompany(updatedInfo.getCompany());
        existingInfo.setAddress(updatedInfo.getAddress());
        existingInfo.setRequirements(updatedInfo.getRequirements());

        existingInfo.setStatus(ClientStatus.READY_FOR_QA);
        return clientRepository.save(existingInfo);
    }

    @Tool(name="qaVerify",description = "complete QA verification task")
    @Transactional
    public ClientInfo qaVerify(ClientInfo clientInfo) {
        ClientInfo existingInfo = clientRepository.findById(clientInfo.getId())
                .orElseThrow(() -> new ClientNotFoundException("Client not found"));

        existingInfo.setQaStatus(clientInfo.getQaStatus());

        if (clientInfo.getQaStatus() == QAStatus.APPROVED) {
            existingInfo.setStatus(ClientStatus.READY_FOR_APPROVAL);
        } else if (clientInfo.getQaStatus() == QAStatus.REJECTED) {
            existingInfo.setStatus(ClientStatus.INFORMATION_GATHERING);
        }
        return clientRepository.save(existingInfo);
    }

    @Tool(name="approve",description = "Case approval task for approver")
    @Transactional
    public ClientInfo approve(ClientInfo clientInfo) {
        ClientInfo existingInfo = clientRepository.findById(clientInfo.getId())
                .orElseThrow(() -> new ClientNotFoundException("Client not found"));

        existingInfo.setApprovalStatus(clientInfo.getApprovalStatus());

        if (clientInfo.getApprovalStatus() == ApprovalStatus.APPROVED) {
            existingInfo.setStatus(ClientStatus.APPROVED);
        } else if (clientInfo.getApprovalStatus() == ApprovalStatus.REJECTED) {
            existingInfo.setStatus(ClientStatus.READY_FOR_QA);
            existingInfo.setQaStatus(QAStatus.NOT_STARTED);
        }

        return clientRepository.save(existingInfo);
    }

    @Tool(name="getClientInfo",description = "get client details by id")
    public ClientInfo getClientInfo(String id) {
        return clientRepository.findById(id)
                .orElseThrow(() -> new ClientNotFoundException("Client not found"));
    }

    public List<ClientInfo> getClientsByStatus(ClientStatus status) {
        return clientRepository.findAll();
    }
}
