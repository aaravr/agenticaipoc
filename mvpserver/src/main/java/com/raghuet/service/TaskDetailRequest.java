package com.raghuet.service;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class TaskDetailRequest {
    private String taskId;
}
