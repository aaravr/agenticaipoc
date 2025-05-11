package com.raghuet.service;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Builder
@Data
@Getter
@Setter
public class TaskDetail {
    private String taskName;
    private String taskId;
    private String taskKey;
    private String status;
}
