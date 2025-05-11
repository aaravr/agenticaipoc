package com.raghuet.service;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class CompleteRequest {
    private String taskId;
    private String taskKey;
    private CompletePayload taskAction;
    private boolean skipBsOutreach;
    private String skipBsOutreachComment;
}
