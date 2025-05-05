package com.raghuet.model;

import lombok.Data;
import jakarta.persistence.*;

import java.time.LocalDateTime;

@Data
@Entity
public class ClientInfo {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    private String name;
    private String email;
    private String phone;
    private String company;
    private String address;
    private String requirements;

    @Enumerated(EnumType.STRING)
    private ClientStatus status;

    @Enumerated(EnumType.STRING)
    private QAStatus qaStatus;

    @Enumerated(EnumType.STRING)
    private ApprovalStatus approvalStatus;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

}