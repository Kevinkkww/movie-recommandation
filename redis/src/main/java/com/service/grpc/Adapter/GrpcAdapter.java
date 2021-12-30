package com.service.grpc.Adapter;

import java.util.Map;

public interface GrpcAdapter {
    Map.Entry<byte[], Object> getInstance(String[] items);

}
