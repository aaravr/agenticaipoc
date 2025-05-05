package com.raghuet;

import com.raghuet.model.ApprovalStatus;
import com.raghuet.model.ClientInfo;
import com.raghuet.model.ClientStatus;
import com.raghuet.model.QAStatus;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
public class ClientRepository {
    List<ClientInfo> findByStatus(ClientStatus status) {
        return Collections.emptyList();
    }

    List<ClientInfo> findByQaStatus(QAStatus qaStatus) {
        return Collections.emptyList();
    }

    List<ClientInfo> findByApprovalStatus(ApprovalStatus approvalStatus) {
        return Collections.emptyList();
    }

    public ClientInfo save(ClientInfo clientInfo) {
       clientInfo.setId(UUID.randomUUID().toString());
       return clientInfo;
    }

    public Optional<ClientInfo> findById(String id) {
        ClientInfo clientInfo =  new ClientInfo();
        clientInfo.setId(id);
        clientInfo.setStatus(ClientStatus.NEW);
        clientInfo.setEmail("test@email.com");
        clientInfo.setName("Test User");
        clientInfo.setCompany("Abc Limited");
        clientInfo.setPhone("07323131312321");
        clientInfo.setQaStatus(QAStatus.NOT_STARTED);
        return Optional.of(clientInfo);
    }


    public List<ClientInfo> findAll() {
        return Collections.emptyList();
    }
}
