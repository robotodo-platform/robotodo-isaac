
import torch

import pxr



# TODO rm
def _todo_contact(contact_headers, contact_datas, friction_anchors = None):

    for contact_header in contact_headers:

        path_entity0 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0)
        path_entity1 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1)

        match contact_header.type:
            case contact_header.type.CONTACT_FOUND | contact_header.type.CONTACT_PERSIST:
                pass
            case contact_header.type.CONTACT_LOST:
                # TODO skip 
                pass
            case _:
                # TODO
                raise RuntimeError("TODO")

        # TODO 
        # actor_pair = (pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
        # collider_pair = (pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1))



        contact_data_offset = contact_header.contact_data_offset
        num_contact_data = contact_header.num_contact_data

        positions = []
        impulses = []
        normals = []
        separations = []

        for i in range(contact_data_offset, contact_data_offset + num_contact_data):
            contact_data = contact_datas[i]
            # TODO ref https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.ContactData
            # TODO this belongs to the header???
            # pxr.PhysicsSchemaTools.intToSdfPath(contact_data.material0), pxr.PhysicsSchemaTools.intToSdfPath(contact_data.material1)
            # TODO only valid for mesh; necesito? positions should be enough?>??
            # contact_data.face_index0, contact_data.face_index1
            normals.append(contact_data.normal)
            impulses.append(contact_data.impulse)
            positions.append(contact_data.position)
            separations.append(contact_data.separation)

        contact_point = ContactPoint(
            position=torch.asarray(positions),
            impulse=torch.asarray(impulses),
            normal=torch.asarray(normals),
            separation=torch.asarray(separations),
        )


        impulses = []
        positions = []

        if friction_anchors is not None:
            friction_anchors_offset = contact_header.friction_anchors_offset
            num_friction_anchors_data = contact_header.num_friction_anchors_data

            for i in range(friction_anchors_offset, friction_anchors_offset + num_friction_anchors_data):
                friction_anchor = friction_anchors[i]
                # TODO ref https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.FrictionAnchor
                impulses.append(friction_anchor.impulse)
                positions.append(friction_anchor.position)

        contact_anchor = ContactAnchor(
            position=torch.asarray(positions),
            impulse=torch.asarray(impulses),
        )

        contact = Contact(
            # TODO FIXME
            entity0=path_entity0,
            entity1=path_entity1,
            #
            points=contact_point,
            anchors=contact_anchor,
        )

        # TODO


import pxr


print(pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
print(pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1))
print(contact_header.type)

dict(
    type=contact_header.type,
    actor_pair=(pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.actor1)),
    collider_pair=(pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0), pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1)),
)




@scene._isaac_physx_simulation.subscribe_full_contact_report_events
def contact_report_sub(contact_headers, contact_datas, friction_anchors):
    pass


class _IsaacContactReportHandler():
    def __init__(self):
        self._subscriptions = dict()

    def subscribe(self, prim_path: str, listener: ...):
        pass

    def __call__(
        self,
        contact_headers, 
        contact_datas, 
        friction_anchors = None,
    ):
        # TODO
        import pxr

        for contact_header in contact_headers:
            path_entity0 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider0)
            path_entity1 = pxr.PhysicsSchemaTools.intToSdfPath(contact_header.collider1)



# TODO necesito???
_ = """

import abc
from typing import Collection

from robotodo.utils.event import BaseAsyncEventStream


class BaseContactCollection(Collection[tuple[Entity, Entity]]):
    on_create: BaseAsyncEventStream[tuple[Entity, Entity]]
    on_update: BaseAsyncEventStream[tuple[Entity, Entity]]
    on_remove: BaseAsyncEventStream[tuple[Entity, Entity]]

    @abc.abstractmethod
    def __contains__(self, item: tuple[Entity, Entity]):
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def __len__(self):
        ...


"""