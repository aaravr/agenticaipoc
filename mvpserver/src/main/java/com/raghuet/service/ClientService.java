package com.raghuet.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.raghuet.ClientRepository;
import com.raghuet.exception.ClientNotFoundException;
import com.raghuet.model.ApprovalStatus;
import com.raghuet.model.ClientInfo;
import com.raghuet.model.ClientStatus;
import com.raghuet.model.QAStatus;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.UUID;

@Service
public class ClientService {
    private static final Logger logger = LoggerFactory.getLogger(ClientService.class);

//    private final RestTemplate restTemplate;
//    private final String BASE_URL = "https://dummyjson.com";

    @Autowired
    private ClientRepository clientRepository;

    List<String> tasks =
            List.of("TASK_1_010", "TASK_1_035", "TASK_1_040", "TASK_1_050", "TASK_1_070", "TASK_1_085", "TASK_1_090");

    List<TaskDetail>  details =  tasks.stream().map(s -> TaskDetail.builder()
            .taskId(UUID.randomUUID().toString())
            .taskKey(s).status("Unavailable").build()).toList();

    @Tool(name="saveClientInfo", description = "Save or create new client details")
//    @Transactional
    public ClientInfo saveClientInfo(ClientInfo clientInfo) {
        logger.info("Received clientInfo: {}", clientInfo);
        if (clientInfo == null) {
            logger.error("clientInfo is null!");
            return null;
        }
        logger.info("clientInfo fields: name={}, email={}, phone={}, company={}", 
                   clientInfo.getName(), clientInfo.getEmail(), 
                   clientInfo.getPhone(), clientInfo.getCompany());
        
        clientInfo.setStatus(ClientStatus.READY_FOR_QA);
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
        clientRepository.save(existingInfo);
        return existingInfo;
    }

    @Tool(name="qaVerify", description = "complete QA verification task")
    @Transactional
    public ClientInfo qaVerify(ClientInfo clientInfo) {
        ClientInfo existingInfo = clientRepository.findById(clientInfo.getId())
                .orElseThrow(() -> new ClientNotFoundException("Client not found"));

        //existingInfo.setQaStatus(clientInfo.getQaStatus());
        clientInfo.setApprovalStatus(ApprovalStatus.QA_VERIFIED);

//       if (clientInfo.getQaStatus() == QAStatus.APPROVED) {
//             existingInfo.setStatus(ClientStatus.READY_FOR_APPROVAL);
//        } else if (clientInfo.getQaStatus() == QAStatus.REJECTED) {
//            existingInfo.setStatus(ClientStatus.INFORMATION_GATHERING);
//        }
        // clientRepository.save(existingInfo);
        return clientInfo;
    }

    @Tool(name="approve", description = "Case approval task for approver")
    @Transactional
    public ClientInfo approve(ClientInfo clientInfo) {
        ClientInfo existingInfo = clientRepository.findById(clientInfo.getId())
                .orElseThrow(() -> new ClientNotFoundException("Client not found"));

        clientInfo.setApprovalStatus(ApprovalStatus.QA_VERIFIED);

        if (clientInfo.getApprovalStatus() == ApprovalStatus.APPROVED) {
            existingInfo.setStatus(ClientStatus.APPROVED);
        } else if (clientInfo.getApprovalStatus() == ApprovalStatus.REJECTED) {
            existingInfo.setStatus(ClientStatus.READY_FOR_QA);
            existingInfo.setQaStatus(QAStatus.NOT_STARTED);
        }
        existingInfo.setApprovalStatus(ApprovalStatus.APPROVED);

        clientRepository.save(existingInfo);
        return clientInfo;
    }

    @Tool(name="getClientInfo", description = "get client details by id")
    public ClientInfo getClientInfo(String id) {
        logger.info("getClientInfo called with id: {}", id);
        return clientRepository.findById(id)
                .orElseThrow(() -> new ClientNotFoundException("Client not found"));
    }

    @Tool(name = "getTaskDetails", description = "get task details")
    public List<TaskDetail> getTaskDetails(TaskDetailRequest taskDetailRequest) throws JsonProcessingException {
        return getTaskDetails();
    }

    @Tool(name = "claimTask", description = "claim a task")
    public void claimTask(ClaimRequest claimRequest) {
        logger.info(" Task id for claiming {}", claimRequest.taskId);
    }

    @Tool(name = "completeTask", description = "complete a task")
    public void completeTask(CompleteRequest completeRequest) {
        logger.info(" Task id for completed {}", completeRequest.getTaskId());
        for (TaskDetail task : details) {
            if ("Open".equals(task.getStatus()) && task.getTaskId().equals(completeRequest.getTaskId())) {
                task.setStatus("Complete");
                break; // stop after updating the first one
            }
        }
    }

    public List<ClientInfo> getClientsByStatus(ClientStatus status) {
        return clientRepository.findAll();
    }


    private List<TaskDetail> getTaskDetails() {
        for (TaskDetail task : details) {
            if ("Open".equals(task.getStatus())) {
                return details;
            } else if ("Unavailable".equals(task.getStatus())) {
                task.setStatus("Open");
                return details;
            }
        }
        return details;
    }
}
