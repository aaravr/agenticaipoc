package com.raghuet.service;


import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ClaimRequest {
    String taskId;
}
