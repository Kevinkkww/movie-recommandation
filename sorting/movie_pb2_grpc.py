# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import movie_pb2 as movie__pb2


class GreeterStub(object):
    """The greeting service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getMovie = channel.unary_unary(
                '/redisData.Greeter/getMovie',
                request_serializer=movie__pb2.MovieRequest.SerializeToString,
                response_deserializer=movie__pb2.MovieReply.FromString,
                )


class GreeterServicer(object):
    """The greeting service definition.
    """

    def getMovie(self, request, context):
        """Sends a greeting
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GreeterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getMovie': grpc.unary_unary_rpc_method_handler(
                    servicer.getMovie,
                    request_deserializer=movie__pb2.MovieRequest.FromString,
                    response_serializer=movie__pb2.MovieReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'redisData.Greeter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Greeter(object):
    """The greeting service definition.
    """

    @staticmethod
    def getMovie(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/redisData.Greeter/getMovie',
            movie__pb2.MovieRequest.SerializeToString,
            movie__pb2.MovieReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
